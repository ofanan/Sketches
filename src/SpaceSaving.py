import matplotlib 
import matplotlib.pyplot as plt
import math, random, os, pickle, mmh3, time, heapq
import numpy as np
from datetime import datetime
from collections import defaultdict
import settings, PerfectCounter, Buckets, NiceBuckets, SEAD_stat, SEAD_dyn, F2P_li, F2P_si, Morris, CEDAR
from settings import warning, error, INF_INT, VERBOSE_RES, VERBOSE_PCL, VERBOSE_LOG, VERBOSE_DETAILED_LOG, VERBOSE_LOG_END_SIM, VERBOSE_LOG_DWN_SMPL, calcPostSimStat
from settings import getRelativePathToTraceFile, checkIfInputFileExists
from   tictoc import tic, toc # my modules for measuring and print-out the simulation time.
from printf import printf, printarFp 
from SingleCntrSimulator import getFxpCntrMaxVal, genCntrMasterFxp
from CountMinSketch import CountMinSketch
from _ast import Or

class SpaceSaving (CountMinSketch):

    # Generate a string that details the parameters' values.
    genSettingsStr = lambda self : f'ss_{self.traceFileName}_{self.mode}_n{self.cntrSize}'
    
    def __init__(self,
        mode            = None, # the counters' mode (e.g., SEC, AEE, realCounter).
        cntrSize        = 8, # num of bits in each counter
        verbose         = [], # The chosen verbose options, detailed in settings.py, determine the output (e.g., to a .pcl, .res or .log file).
        cacheSize       = 1, # number of counters -- actually, the cache's size
        numFlows        = 10, # the total number of flows to be estimated.
        seed            = settings.SEED,
        traceFileName   = None,
        maxValBy        = None, # How to calculate the maximum value (for SEAD/CEDAR).   
        numOfExps       = 1, 
        maxNumIncs      = INF_INT, # maximum # of increments (pkts in the trace), after which the simulation will be stopped. 
    ):
        
        """
        """
        self.cntrSize, self.traceFileName = cntrSize, traceFileName
        self.maxNumIncs, self.numOfExps, = maxNumIncs, numOfExps
        self.numCntrs, self.numFlows, self.mode, self.seed = cacheSize, numFlows, mode, seed
        self.verbose = verbose
        self.cntrsAr = defaultdict(int)
        self.genOutputDirectories ()
        self.openOutputFiles ()
        self.flowIds    = [None]*self.numCntrs
        self.flowSizes  = np.zeros(self.numCntrs)
        self.dwnSmpl         = self.mode.endswith('_ds')
        self.maxValBy        = maxValBy
        if self.maxValBy==None: # By default, the maximal counter's value is the trace length 
            if self.traceFileName=='Rand':
                self.cntrMaxVal = self.maxNumIncs
            else:
                self.cntrMaxVal = settings.getTraceLen(self.traceFileName)
        else:
            self.cntrMaxVal = getFxpCntrMaxVal (cntrSize=self.cntrSize, fxpSettingStr=self.maxValBy)
        random.seed (self.seed)

    def incNQueryFlow(
            self, 
            flowId
        ):
        """
        Update the value for a single flow. Return the updated estimated value for this flow.
        To ease the finding of min item (without the need to perform cntr2num), we cache also the cached values.  
        """
        hit = False
        for cntrIdx in range(self.numCntrs): # loop over the cache's elements
            if self.flowIds[cntrIdx]==flowId: # found the flowId in the $
                self.flowSizes[cntrIdx] = self.cntrMaster.incCntrBy1GetVal (cntrIdx=cntrIdx) # prob-inc. the counter, and get its val
                hit = True # $ hit
                break
            elif self.flowIds[cntrIdx]==None: # the flowId isn't cached yet, and the $ is not full yet
                self.flowIds  [cntrIdx] = flowId # insert flowId into the $
                self.flowSizes[cntrIdx] = self.cntrMaster.incCntrBy1GetVal (cntrIdx=cntrIdx) # prob-inc. the counter, and get its val
                hit = True # $ hit
                break
        if not(hit): # didn't found flowId in the $ --> insert it
            cntrIdx = min(range(self.numCntrs), key=self.flowSizes.__getitem__) # find the index of the minimal cached item # to allow randomizing between all minimal items, np.where(a==a.min())
            self.flowIds  [cntrIdx] = flowId # replace the item by the newly-inserted flowId
            self.flowSizes[cntrIdx] = self.cntrMaster.incCntrBy1GetVal (cntrIdx=cntrIdx) # prob'-inc. the value
        return self.flowSizes[cntrIdx]
        
    def openOutputFiles (self) -> None:
        """
        Open the output files (.res, .log, .pcl), as defined by the verbose level requested.
        """      
        if VERBOSE_PCL in self.verbose:
            self.pclOutputFile = open(f'../res/pcl_files/ss_{self.traceFileName}_{settings.getMachineStr()}.pcl', 'ab+')

        if (VERBOSE_RES in self.verbose):
            self.resFile = open (f'../res/ss_{self.traceFileName}_{settings.getMachineStr()}.res', 'a+')
            
        if (settings.VERBOSE_FULL_RES in self.verbose):
            self.fullResFile = open (f'../res/ss_M{self.numCntrs}_{settings.getMachineStr()}_full.res', 'a+')

        self.logFile =  None # default
        if VERBOSE_LOG in self.verbose or \
           VERBOSE_DETAILED_LOG in self.verbose or\
           VERBOSE_LOG_DWN_SMPL in self.verbose:
            self.logFile = open (f'../res/log_files/{self.genSettingsStr()}.log', 'w')
    
    def printSimMsg (self, str):
        """
        Print-screen an info msg about the parameters and hours of the simulation starting to run. 
        """             
        print ('{} running sim at t={}. trace={}, numOfExps={}, mode={}, cntrSize={}, cacheSize={}' .format (
                        str, datetime.now().strftime('%H:%M:%S'), self.traceFileName, self.numOfExps, self.mode, self.cntrSize, self.numCntrs))


    def printLogLine (
            self, 
            flowId, 
            estimatedVal, 
            realVal):
        """
        Print a log line to the logFile during simulation
        """
        if not(VERBOSE_LOG in self.verbose):
            return
        self.cntrMaster.printAllCntrs (self.logFile)
        printf (self.logFile, 
                ' incNum={}, flowIds={}, flowSizes={}, flowId={}, estimatedVal={:.0f} realVal={:.0f}' .format(
                self.incNum, self.flowIds, self.flowSizes, flowId, estimatedVal, realVal)) 

    def runSimFromTrace (self):
        """
        Run a simulation where the input is taken from self.traceFileName.
        """

        if self.numFlows==None:
            error ('In SpaceSaving.runSimFromTrace(). Sorry, dynamically calculating the flowNum is not supported yet.')

        relativePathToInputFile = settings.getRelativePathToTraceFile (f'{self.traceFileName}.txt')
        checkIfInputFileExists (relativePathToInputFile)
        for self.expNum in range (self.numOfExps):
            self.seed = self.expNum+1 
            random.seed (self.seed)
            self.genCntrMaster () # Generate a fresh, empty CntrMaster, for each experiment
            self.cntrMaster.setLogFile(self.logFile)
            flowRealVal = [0] * self.numFlows
            self.incNum = 0
            self.writeProgress () # log the beginning of the experiment; used to track the progress of long runs.
            
            traceFile = open (relativePathToInputFile, 'r')
            for row in traceFile:            
                flowId = int(row) 
                self.incNum  += 1                
                flowRealVal[flowId]     += 1
                
                flowEstimatedVal   = self.incNQueryFlow (flowId=flowId)
                sqEr = (flowRealVal[flowId] - flowEstimatedVal)**2
                self.sumSqAbsEr[self.expNum] += sqEr    
                self.sumSqRelEr[self.expNum] += sqEr/(flowRealVal[flowId])**2                
                self.printLogLine (
                    flowId          = flowId, 
                    estimatedVal    = flowEstimatedVal,
                    realVal         = flowRealVal[flowId]
                )
                if VERBOSE_DETAILED_LOG in self.verbose and self.incNum>10000: #$$$
                    printf (self.logFile, 'incNum={}, realVal={}, estimated={:.1e}, sqAbsEr={:.1e}, sqRelEr={:.1e}, sumAbsSqEr={:.1e}, sumRelSqEr={:.1e}\n' .format (self.incNum, flowRealVal[flowId], flowEstimatedVal, sqEr, sqEr/(flowRealVal[flowId])**2, self.sumSqAbsEr[self.expNum], self.sumSqRelEr[self.expNum]))
                if self.incNum==self.maxNumIncs:
                    break
        traceFile.close ()
    
        if VERBOSE_LOG_END_SIM in self.verbose:
            self.cntrMaster.printCntrsStat (self.logFile, genPlot=True, outputFileName=self.genSettingsStr()) 
            self.cntrMaster.printAllCntrs  (self.logFile)
    
    def runSimRandInput (self):
        """
        Run a simulation with synthetic, randomly-generated, input.
        """
             
        randInput = True
        for self.expNum in range (self.numOfExps):
            flowRealVal = [0] * self.numFlows
            self.writeProgress () # log the beginning of the experiment; used to track the progress of long runs.
            self.genCntrMaster ()
            self.setLogFile (self.logFile)

            for self.incNum in range(self.maxNumIncs):
                flowId = random.randint (0, self.numFlows-1)
                flowRealVal[flowId]     += 1
                flowEstimatedVal   = self.incNQueryFlow (flowId=flowId)
                sqEr = (flowRealVal[flowId] - flowEstimatedVal)**2
                self.sumSqAbsEr[self.expNum] += sqEr                
                self.sumSqRelEr[self.expNum] += sqEr/(flowRealVal[flowId])**2                
                self.printLogLine (
                    flowId          = flowId, 
                    estimatedVal    = flowEstimatedVal,
                    realVal         = flowRealVal[flowId]
                )     
            if settings.VERBOSE_FULL_RES in self.verbose:
                dict = settings
            
    def sim (
        self, 
        ):
        """
        Simulate the Space Saving cache.
        """
        
        self.sumSqAbsEr  = [0] * self.numOfExps # self.sumSqAbsEr[j] will hold the sum of the square absolute errors collected at experiment j. 
        self.sumSqRelEr  = [0] * self.numOfExps # self.sumSqRelEr[j] will hold the sum of the square relative errors collected at experiment j.        
        self.printSimMsg ('Started')
        self.openOutputFiles ()
        tic ()
        if self.traceFileName=='Rand': # random input
            self.runSimRandInput ()
        else: # read trace from a file
            self.runSimFromTrace ()
        toc ()
        for rel_abs_n in [True, False]:
            for statType in ['Mse', 'normRmse']:
                sumSqEr = self.sumSqRelEr if rel_abs_n else self.sumSqAbsEr
                dict = calcPostSimStat(
                    sumSqEr         = sumSqEr, 
                    statType        = statType, 
                    numMeausures    = self.incNum+1,
                    verbose         = self.verbose,
                    logFile         = self.logFile
                    )
                dict = self.fillStatDictsFields(dict)
                dict['rel_abs_n']   = rel_abs_n
    
                if VERBOSE_PCL in self.verbose:
                    self.dumpDictToPcl    (dict)
                if VERBOSE_RES in self.verbose:
                    printf (self.resFile, f'{dict}\n\n') 
        self.printSimMsg (f'Finished {self.incNum+1} increments')

                
    def fillStatDictsFields (self, dict) -> dict:
        """
        Add to the given dict some fields detailing the sim settings. Return the full dict.
        """
        dict['numOfExps']   = self.expNum+1# The count of the experiments started in 0
        dict['numIncs']     = self.incNum
        dict['mode']        = self.mode
        dict['cacheSize']   = self.numCntrs
        dict['numFlows']    = self.numFlows
        return dict
    
def runSS (mode, 
    cntrSize    = 8,
    maxNumIncs  = float ('inf'),
    cacheSize   = 1,
    traceFileName = 'Rand' 
):
    """
    """   
    if traceFileName=='Rand':
        ss = SpaceSaving (
            numFlows        = 9,
            cntrSize        = cntrSize, 
            cacheSize       = 3,
            verbose         = [VERBOSE_LOG_DWN_SMPL], # VERBOSE_LOG, VERBOSE_LOG_END_SIM, VERBOSE_LOG, settings.VERBOSE_DETAILS
            traceFileName   = traceFileName,
            mode            = mode,
            numOfExps       = 1, 
            maxNumIncs      = 2222
        )
        ss.sim ()
    else:
        ss = SpaceSaving (
            cntrSize        = cntrSize,
            numFlows        = settings.getNumFlowsByTraceName (traceFileName), 
            cacheSize       = cacheSize,
            verbose         = [VERBOSE_RES, VERBOSE_PCL], #$$$ [VERBOSE_RES, VERBOSE_PCL] # VERBOSE_LOG_END_SIM,  VERBOSE_RES, settings.VERBOSE_FULL_RES, VERBOSE_PCL] # VERBOSE_LOG, VERBOSE_RES, VERBOSE_PCL, settings.VERBOSE_DETAILS
            mode            = mode,
            traceFileName   = traceFileName,
            numOfExps       = 10, 
        )
        ss.sim ()
    
if __name__ == '__main__':
    try:
        for cacheSize in [2]: #[2**i for i in range(10, 19)]:
            # for mode in ['F2P_li_h2', 'F3P_li_h3']:    
            for mode in ['CEDAR_ds']:    
                runSS (
                    cntrSize        = 4,
                    mode            = mode,
                    cacheSize       = cacheSize,
                    traceFileName   = 'Rand',
                )
    except KeyboardInterrupt:
        print('Keyboard interrupt.')
