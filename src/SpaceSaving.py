import matplotlib 
import matplotlib.pyplot as plt
import math, random, os, pickle, mmh3, time, heapq
import numpy as np
from datetime import datetime
import settings, PerfectCounter, Buckets, NiceBuckets, SEAD_stat, SEAD_dyn, F2P_li, F2P_si, Morris, CEDAR
from settings import warning, error, VERBOSE_RES, VERBOSE_PCL, VERBOSE_LOG, VERBOSE_DETAILED_LOG, VERBOSE_LOG_END_SIM, calcPostSimStat
from   tictoc import tic, toc # my modules for measuring and print-out the simulation time.
from printf import printf, printarFp
from SingleCntrSimulator import getFxpCntrMaxVal, genCntrMasterFxp
from CountMinSketch import CountMinSketch

class SpaceSaving (CountMinSketch):

    # Generate a string that details the parameters' values.
    genSettingsStr = lambda self : f'ss_{self.traceFileName}_{self.mode}_bit{self.cntrSize}'
    
    def printCahe (self):
        """
        Used for debugginign only.
        print the id, level and CPU of each user in a heap.
        """
        for item in self.cache:
            print (f'key = {item.key}, val = {item.value}')
        print ('')
        
    def __init__(self,
        mode            = None, # the counter mode (e.g., SEC, AEE, realCounter).
        cntrSize        = 8, # num of bits in each counter
        verbose         = [], # The chosen verbose options, detailed in settings.py, determine the output (e.g., to a .pcl, .res or .log file).
        cacheSize       = 1, # number of counters -- actually, the cache's size
        numFlows        = 10, # the total number of flows to be estimated.
        seed            = settings.SEED,
    ):
        
        """
        """
        self.cntrSize   = cntrSize
        self.cacheSize, self.numFlows, self.mode, self.seed = cacheSize, numFlows, mode, seed
        self.numCntrs = self.cacheSize
        self.verbose = verbose
        self.genOutputDirectories ()
        # To ease the finding of min item (without the need to perform cntr2num), we cache also the cached values.  
        self.cache = [{'flowId' : None, 'val' : 0}]*self.cacheSize

    def incNQueryFlow(
            self, 
            flowId
        ):
        """
        Update the value for a single flow. Return the updated estimated value for this flow.
        To ease the finding of min item (without the need to perform cntr2num), we cache also the cached values.  
        """
        cntrIdx = flowId%self.cacheSize
        if self.cache[cntrIdx]['flowId'] == flowId: # the item is already cached
            self.cache[cntrIdx]['val'] = self.cntrMaster.incCntrBy1GetVal (cntrIdx=cntrIdx) # prob-inc. the counter, and get its val  
        else: # the item is not cached yet
            idxOfMinCntr = min(range(self.cacheSize), key=[item['val'] for item in self.cache].__getitem__) # find the index of the minimal cached item
            self.cache[cntrIdx]['flowId'] = flowId # replace the item by the newly-inserted flowId
            self.cache[cntrIdx]['val'] = self.cntrMaster.incCntrBy1GetVal (cntrIdx=idxOfMinCntr) # prob'-inc. the value
        return self.cache[cntrIdx]['val']
   
    def openOutputFiles (self) -> None:
        """
        Open the output files (.res, .log, .pcl), as defined by the verbose level requested.
        """      
        if VERBOSE_PCL in self.verbose:
            self.pclOutputFile = open(f'../res/pcl_files/ss_M{self.cacheSize}_{settings.getMachineStr()}.pcl', 'ab+')

        if (VERBOSE_RES in self.verbose):
            self.resFile = open (f'../res/ss_M{self.cacheSize}__{settings.getMachineStr()}.res', 'a+')
            
        if (settings.VERBOSE_FULL_RES in self.verbose):
            self.fullResFile = open (f'../res/ss_M{self.cacheSize}_{settings.getMachineStr()}_full.res', 'a+')

        self.logFile =  None # default
        if VERBOSE_LOG in self.verbose or VERBOSE_DETAILED_LOG in self.verbose:
            self.logFile = open (f'../res/log_files/{self.genSettingsStr()}.log', 'w')
            
    def printSimMsg (self, str):
        """
        Print-screen an info msg about the parameters and hours of the simulation starting to run. 
        """             
        print ('{} running sim at t={}. trace={}, numOfExps={}, mode={}, cntrSize={}, cacheSize={}' .format (
                        str, datetime.now().strftime('%H:%M:%S'), self.traceFileName, self.numOfExps, self.mode, self.cntrSize, self.cacheSize))

    def runSimFromTrace (self):
        """
        Run a simulation where the input is taken from self.traceFileName.
        """

        if self.numFlows==None:
            error ('In SpaceSaving.runSimFromTrace(). Sorry, dynamically calculating the flowNum is not supported yet.')
        self.genCntrMaster ()
        if (VERBOSE_LOG in self.verbose) or (VERBOSE_LOG_END_SIM in self.verbose):
            infoStr = '{}_{}' .format (self.genSettingsStr(), self.cntrMaster.genSettingsStr())
            self.logFile = open (f'../res/log_files/{infoStr}.log', 'w')
            self.cntrMaster.setLogFile(self.logFile)

        relativePathToInputFile = settings.getRelativePathToTraceFile (self.traceFileName)
        for self.expNum in range (self.numOfExps):
            self.seed = self.expNum+1 
            random.seed (self.seed)
            self.genCntrMaster () # Generate a fresh, empty CntrMaster, for each experiment
            flowRealVal = [0] * self.numFlows
            self.incNum = 0
            self.writeProgress () # log the beginning of the experiment; used to track the progress of long runs.

            if (VERBOSE_LOG in self.verbose) or (settings.VERBOSE_PROGRESS in self.verbose) or (VERBOSE_LOG_END_SIM in self.verbose):
                infoStr = '{}_{}' .format (self.genSettingsStr(), self.cntrMaster.genSettingsStr())
                self.logFile = open (f'../res/log_files/{infoStr}.log', 'a+')
                self.cntrMaster.setLogFile(self.logFile)
            
            traceFile = open (relativePathToInputFile, 'r')
            for row in traceFile:            
                flowId = int(row) 
                self.incNum  += 1                
                flowRealVal[flowId]     += 1
                
                flowEstimatedVal   = self.incNQueryFlow (flowId=flowId)
                sqEr = (flowRealVal[flowId] - flowEstimatedVal)**2
                self.sumSqAbsEr[self.expNum] += sqEr    
                self.sumSqRelEr[self.expNum] += sqEr/(flowRealVal[flowId])**2                
                if VERBOSE_LOG in self.verbose:
                    self.cntrMaster.printAllCntrs (self.logFile)
                    printf (self.logFile, 'incNum={}, hashes={}, estimatedVal={:.0f} realVal={:.0f} \n' .format(self.incNum, flowId%self.cacheSize, flowEstimatedVal, flowRealVal[flowId])) 
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
             
        self.traceFileName  = 'rand'
        
        randInput = True
        for self.expNum in range (self.numOfExps):
            flowRealVal = [0] * self.numFlows
            self.writeProgress () # log the beginning of the experiment; used to track the progress of long runs.
            self.genCntrMaster ()

            if (VERBOSE_LOG in self.verbose) or (settings.VERBOSE_PROGRESS in self.verbose) or (VERBOSE_LOG_END_SIM in self.verbose):
                infoStr = '{}_{}' .format (self.genSettingsStr(), self.cntrMaster.genSettingsStr())
                self.logFile = open (f'../res/log_files/{infoStr}.log', 'a+')
                self.cntrMaster.setLogFile(self.logFile)
            
            for self.incNum in range(self.maxNumIncs):
                flowId = math.floor(np.random.exponential(scale = 2*math.sqrt(self.numFlows))) % self.numFlows
                flowRealVal[flowId]     += 1
                flowEstimatedVal   = self.incNQueryFlow (flowId=flowId)
                sqEr = (flowRealVal[flowId] - flowEstimatedVal)**2
                self.sumSqAbsEr[self.expNum] += sqEr                
                self.sumSqRelEr[self.expNum] += sqEr/(flowRealVal[flowId])**2                
                if VERBOSE_LOG in self.verbose:
                    self.cntrMaster.printAllCntrs (self.logFile)
                    printf (self.logFile, 'incNum={}, hashes={}, estimatedVal={:.0f} realVal={:.0f} \n' .format(self.incNum, flowId%self.cacheSize, flowEstimatedVal, flowRealVal[flowId])) 
            if settings.VERBOSE_FULL_RES in self.verbose:
                dict = settings
            
    def sim (
        self, 
        maxNumIncs     = 5000, # maximum # of increments (pkts in the trace), after which the simulation will be stopped. 
        numOfExps      = 1,  # number of repeated experiments. Relevant only for randomly-generated traces.
        traceFileName  = None
        ):
        """
        Simulate the count min sketch
        """
        
        self.maxNumIncs, self.numOfExps, self.traceFileName = maxNumIncs, numOfExps, traceFileName
        self.sumSqAbsEr  = [0] * self.numOfExps # self.sumSqAbsEr[j] will hold the sum of the square absolute errors collected at experiment j. 
        self.sumSqRelEr  = [0] * self.numOfExps # self.sumSqRelEr[j] will hold the sum of the square relative errors collected at experiment j.        
        self.printSimMsg ('Started')
        self.openOutputFiles ()
        tic ()
        if self.traceFileName==None: # random input
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
        self.printSimMsg (f'Finished {self.incNum} increments')

                
    def fillStatDictsFields (self, dict) -> dict:
        """
        Add to the given dict some fields detailing the sim settings. Return the full dict.
        """
        dict['numOfExps']   = self.expNum+1# The count of the experiments started in 0
        dict['numIncs']     = self.incNum
        dict['mode']        = self.mode
        dict['cacheSize']   = self.cacheSize
        dict['numFlows']    = self.numFlows
        return dict
    
def runSS (mode, 
    cntrSize    = 8,
    maxNumIncs  = float ('inf'),
    cacheSize   = 1,
    traceFileName = 'Caida1.txt' 
):
    """
    """   
    numFlows = 1276112 # 13,182,023 incs
    
    if traceFileName=='shortTest':
        numFlows      = numFlows
        maxNumIncs    = 20 #4000 #(width * depth * cntrSize**3)/2
        numOfExps     = 2
        verbose       = [VERBOSE_LOG, VERBOSE_RES] # VERBOSE_LOG, VERBOSE_LOG_END_SIM, VERBOSE_LOG, settings.VERBOSE_DETAILS
        traceFileName = None
    else:
        numFlows                = numFlows
        maxNumIncs              = maxNumIncs   
        numOfExps               = 10 #$$$ #100 
        verbose                 = [VERBOSE_RES, VERBOSE_PCL] #$$$ [VERBOSE_RES, VERBOSE_PCL] # VERBOSE_LOG_END_SIM,  VERBOSE_RES, settings.VERBOSE_FULL_RES, VERBOSE_PCL] # VERBOSE_LOG, VERBOSE_RES, VERBOSE_PCL, settings.VERBOSE_DETAILS
    
    ss = SpaceSaving (
        cntrSize    = cntrSize, 
        numFlows    = numFlows, 
        verbose     = verbose,
        mode        = mode,
        cacheSize   = cacheSize,
        )
    ss.sim (
        numOfExps      = numOfExps, 
        maxNumIncs     = maxNumIncs, 
        traceFileName  = traceFileName
        )
    
if __name__ == '__main__':
    try:
        for cacheSize in [3]:
            for mode in ['SEAD_dyn']:    
            # for mode in ['F2P_li_h2']:    
            # for mode in ['F3P_li_h3']:    
            # for mode in ['F2P_lli', 'CEDAR', 'Morris']:    
                runSS (
                    mode          = mode,
                    traceFileName = 'shortTest'
                )
    except KeyboardInterrupt:
        print('Keyboard interrupt.')
