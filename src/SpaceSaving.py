import matplotlib 
import matplotlib.pyplot as plt
import math, random, os, pickle, mmh3, time, heapq
import numpy as np
from datetime import datetime
from collections import defaultdict
import settings, PerfectCounter, Buckets, NiceBuckets, SEAD_stat, SEAD_dyn, F2P_li, F2P_si, Morris, CEDAR
from settings import * 
from ttictoc import tic,toc
from printf import printf, printarFp 
from SingleCntrSimulator import getCntrMaxValFromFxpStr, genCntrMasterFxp
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
        seed            = SEED,
        traceFileName   = None,
        maxValBy        = None, # How to calculate the maximum value (for SEAD/CEDAR).   
        numOfExps       = 1, 
        maxNumIncs      = INF_INT, # maximum # of increments (pkts in the trace), after which the simulation will be stopped. 
    ):
        
        """
        """
        self.cntrSize, self.traceFileName = cntrSize, traceFileName
        self.maxNumIncs, self.numOfExps, = maxNumIncs, numOfExps
        self.cacheSize, self.numFlows, self.mode, self.seed = cacheSize, numFlows, mode, seed
        self.usedCacheSpace = 0 
        self.verbose = verbose
        self.cntrsAr = defaultdict(int)
        self.genOutputDirectories ()
        self.openOutputFiles ()
        self.flowIds    = np.zeros (self.cacheSize, dtype='uint32')
        self.flowSizes  = np.zeros (self.cacheSize, dtype='uint32')
        self.dwnSmpl    = self.mode.endswith('_ds')
        self.maxValBy   = maxValBy
        if self.maxValBy==None: # By default, the maximal counter's value is the trace length 
            if self.traceFileName=='Rand':
                self.cntrMaxVal = self.maxNumIncs
            else:
                self.cntrMaxVal = getTraceLen(self.traceFileName)
        else:
            self.cntrMaxVal = getCntrMaxValFromFxpStr (cntrSize=self.cntrSize, fxpSettingStr=self.maxValBy)
        random.seed (self.seed)

    def incNQueryFlow(
            self, 
            flowId : int, # flow Id to (probabilistic) increment
        ):
        """
        Update the value for a single flow. Return the updated estimated value for this flow.
        To ease the finding of min item (without the need to perform cntr2num), we cache also the cached values.  
        """
        idxOfFlowIdInCache = np.where (self.flowIds==flowId)[0]
        
        if len(idxOfFlowIdInCache)>1:
            error (f'In SpaceSaving.incNQueryFlow(). More than 2 cache entries for flowId {flowId}')
        if len(idxOfFlowIdInCache)==1: # looked item is already cached
            cntrIdx = idxOfFlowIdInCache[0]
            self.flowSizes[cntrIdx] = int(round(self.cntrMaster.incCntrBy1GetVal (cntrIdx=cntrIdx))) # prob-inc. the counter, and get its val
        elif self.usedCacheSpace<self.cacheSize: # looked item is not cached, but cache isn't full
            cntrIdx                  = self.usedCacheSpace
            self.flowIds  [cntrIdx]  = flowId # insert flowId into the $
            self.flowSizes[cntrIdx]  = int(round(self.cntrMaster.incCntrBy1GetVal (cntrIdx=cntrIdx))) # prob-inc. the counter, and get its val
            self.usedCacheSpace     += 1
        else: # didn't find flowId in the $ --> insert the flowId
            cntrIdx = np.argmin (self.flowSizes)
            self.flowIds  [cntrIdx] = flowId # replace the item by the newly-inserted flowId
            self.flowSizes[cntrIdx] = int(round(self.cntrMaster.incCntrBy1GetVal (cntrIdx=cntrIdx))) # prob'-inc. the value
        return self.flowSizes[cntrIdx]
        
    def openOutputFiles (self) -> None:
        """
        Open the output files (.res, .log, .pcl), as defined by the verbose level requested.
        """      
        if VERBOSE_PCL in self.verbose:
            self.pclOutputFile = open(f'../res/pcl_files/ss_{self.traceFileName}_{getMachineStr()}.pcl', 'ab+')

        if (VERBOSE_RES in self.verbose):
            self.resFile = open (f'../res/ss_{self.traceFileName}_{getMachineStr()}.res', 'a+')
            
        if (VERBOSE_FULL_RES in self.verbose):
            self.fullResFile = open (f'../res/ss_M{self.cacheSize}_{getMachineStr()}_full.res', 'a+')

        self.logFile =  None # default
        if VERBOSE_LOG in self.verbose or \
           VERBOSE_DETAILED_LOG in self.verbose or\
           VERBOSE_LOG_DWN_SMPL in self.verbose or\
           VERBOSE_LOG_END_SIM in self.verbose:
            self.logFile = open (f'../res/log_files/{self.genSettingsStr()}.log', 'w')
    
    def printSimMsg (self, str):
        """
        Print-screen an info msg about the parameters and hours of the simulation starting to run. 
        """             
        print ('{} running ss at t={}. trace={}, numOfExps={}, mode={}, cntrSize={}, cacheSize={}' .format (
                        str, datetime.now().strftime('%H:%M:%S'), self.traceFileName, self.numOfExps, self.mode, self.cntrSize, self.cacheSize))


    def printLogLine (
            self, 
            flowId, 
            estimatedVal, 
            realVal
        ):
        """
        Print a log line to the logFile during simulation
        """
        if not(VERBOSE_LOG in self.verbose):
            return
        if realVal%1>0: 
            error (f'In SpaceSaving.printLogLine(). Got realVal={realVal}. The real val of flow size should be an int.')                
        if self.cacheSize < 10:
            self.cntrMaster.printAllCntrs (self.logFile)
        printf (self.logFile, 
                ' incNum={}, flowId={}, flowSizes={}, estimatedVal={:.0f} realVal={}\n' .format(
                self.incNum, flowId, self.flowSizes, estimatedVal, realVal)) 

    def runSimFromTrace (self):
        """
        Run a simulation where the input is taken from self.traceFileName.
        """

        if self.numFlows==None:
            error ('In SpaceSaving.runSimFromTrace(). Sorry, dynamically calculating the flowNum is not supported yet.')

        relativePathToInputFile = getRelativePathToTraceFile (f'{self.traceFileName}.txt')
        checkIfInputFileExists (relativePathToInputFile)
        for self.expNum in range (self.numOfExps):
            self.seed = self.expNum+1 
            random.seed (self.seed)
            self.genCntrMaster () # Generate a fresh, empty CntrMaster, for each experiment
            self.cntrMaster.setLogFile(self.logFile)
            if self.expNum==0 and self.logFile!=None:
                printf (self.logFile, f'// cntrMaxVal = {self.cntrMaxVal}\n')
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
            printf (self.logFile, '// cacheSize={self.cacheSize}')
            self.cntrMaster.printCntrsStat (self.logFile, genPlot=True) 
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
            self.cntrMaster.setLogFile (self.logFile)
            printf (self.logFile, f'// cntrMaxVal = {self.cntrMaxVal}\n')

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
            
    def sim (
        self, 
        ):
        """
        Simulate the count min sketch
        """
        
        self.sumSqAbsEr  = np.zeros (self.numOfExps) # self.sumSqAbsEr[j] will hold the sum of the square absolute errors collected at experiment j. 
        self.sumSqRelEr  = np.zeros (self.numOfExps) # self.sumSqRelEr[j] will hold the sum of the square relative errors collected at experiment j.        
        self.printSimMsg ('Started')
        self.openOutputFiles ()
        tic ()
        if self.numFlows==None:
            error ('In CountMinSketch.runSimFromTrace(). Sorry, dynamically calculating the flowNum is not supported yet.')

        if self.traceName=='Rand': # run synthetic, randomized trace
            rng = np.random.default_rng()
            self.trace = rng.integers (self.numFlows, size=self.maxNumIncs, dtype='uint32')
        else:
            relativePathToInputFile = getRelativePathToTraceFile (f'{getTraceFullName(self.traceName)}.txt')
            checkIfInputFileExists (relativePathToInputFile, exitError=True)
            self.trace = np.fromfile(relativePathToInputFile, count = self.maxNumIncs, sep='\n', dtype='uint32')
        self.maxNumIncs = min (self.maxNumIncs, self.trace.shape[0]) 

        for self.expNum in range (self.numOfExps):
            self.seed = self.expNum+1
            random.seed (self.seed) 
            self.genCntrMaster () # Generate a fresh, empty CntrMaster, for each experiment
            self.cntrMaster.setLogFile(self.logFile)
            flowRealVal = np.zeros(self.numFlows)
            self.writeProgress () # log the beginning of the experiment; used to track the progress of long runs.

            for self.incNum in range(self.maxNumIncs):
                flowId = self.trace[self.incNum]            
                flowRealVal[flowId]     += 1
                flowEstimatedVal = self.incNQueryFlow (flowId)
                sqEr = (flowRealVal[flowId] - flowEstimatedVal)**2
                self.sumSqAbsEr[self.expNum] += sqEr    
                self.sumSqRelEr[self.expNum] += sqEr/(flowRealVal[flowId])**2                
                if VERBOSE_LOG_SHORT in self.verbose: 
                    self.cntrMaster.printAllCntrs (self.logFile, printAlsoVec=False)
                    printf (self.logFile, 'incNum={}, hashes={}, estimatedVal={:.0f} realVal={:.0f} \n' .format(
                        self.incNum, traceHashes[self.incNum], flowEstimatedVal, flowRealVal[flowId]))
                elif VERBOSE_LOG in self.verbose: 
                    self.cntrMaster.printAllCntrs (self.logFile, printAlsoVec=True)
                    printf (self.logFile, 'incNum={}, hashes={}, estimatedVal={:.0f} realVal={:.0f} \n' .format(
                        self.incNum, traceHashes[self.incNum], flowEstimatedVal, flowRealVal[flowId]))
                if VERBOSE_DETAILED_LOG in self.verbose and self.incNum>10000: #$$$
                    printf (self.logFile, 'incNum={}, realVal={}, estimated={:.1e}, sqAbsEr={:.1e}, sqRelEr={:.1e}, sumSqAbsEr={:.1e}, sumSqRelEr={:.1e}\n' .format (
                        self.incNum, flowRealVal[flowId], flowEstimatedVal, sqAbsEr, sqRelEr, self.sumSqAbsEr[self.expNum], self.sumSqRelEr[self.expNum]))
            if self.expNum==0: # Log (if at all) only in the first experiment. No need to log again in further exps.
                self.logEndSim ()
                self.rmvVerboseLogs ()
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
        print (f'Finished {self.incNum+1} increments. {genElapsedTimeStr (toc())}')

                
    def fillStatDictsFields (self, dict) -> dict:
        """
        Add to the given dict some fields detailing the sim settings. Return the full dict.
        """
        dict['numOfExps']   = self.expNum+1# The count of the experiments started in 0
        dict['numIncs']     = self.incNum
        dict['mode']        = self.mode
        dict['cacheSize']   = self.cacheSize
        dict['numFlows']    = self.numFlows
        dict['cntrSize']    = self.cntrSize
        dict['seed']        = self.seed
        dict['maxValBy']    = self.maxValBy
        return dict
 
   
def LaunchSsSim (
        traceFileName   : str, 
        cntrSize        : int, 
        mode            : str, # a string, detailing the mode of the counter, e.g. "F2P_li_h2".
        maxNumIncs      : float = float ('inf'), 
        cacheSize       : int,
    ):
    """
    Lanuch a simulation of Space Saving.
    """
    if traceFileName=='Rand':
        ss = SpaceSaving (
            numFlows        = 9,
            cntrSize        = cntrSize, 
            cacheSize       = 3,
            verbose         = [VERBOSE_LOG, VERBOSE_LOG_DWN_SMPL], # VERBOSE_LOG, VERBOSE_LOG_END_SIM, VERBOSE_LOG, VERBOSE_DETAILS
            traceFileName   = traceFileName,
            mode            = mode,
            numOfExps       = 1, 
            maxNumIncs      = 33,
            maxValBy        = 'F2P_li_h2',
        )
    else:
        ss = SpaceSaving (
            cntrSize        = cntrSize,
            numFlows        = getNumFlowsByTraceName (traceFileName), 
            cacheSize       = cacheSize,
            verbose         = [VERBOSE_RES, VERBOSE_PCL, VERBOSE_LOG_END_SIM, VERBOSE_LOG_DWN_SMPL], # [VERBOSE_RES, VERBOSE_PCL] # VERBOSE_LOG_END_SIM,  VERBOSE_RES, VERBOSE_FULL_RES, VERBOSE_PCL] # VERBOSE_LOG, VERBOSE_RES, VERBOSE_PCL, VERBOSE_DETAILS
            mode            = mode,
            traceFileName   = traceFileName,
            numOfExps       = 10, 
            maxValBy        = 'F2P_li_h2',
        )
    ss.sim ()
    
if __name__ == '__main__':
    try:
        for cacheSize in [2**i for i in range(10, 19)]:
            for traceFileName in ['Caida1', 'Caida2']:
                LaunchSsSim (
                    traceFileName   = traceFileName, 
                    cntrSize        = 8, 
                    mode            = 'F2P_li_h2_ds', # a string, detailing the mode of the counter, e.g. "F2P_li_h2".
                    maxNumIncs      = maxNumIncs, 
                    cacheSize       = cacheSize
                )

    except KeyboardInterrupt:
        print('Keyboard interrupt.')
