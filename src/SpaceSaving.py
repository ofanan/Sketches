import matplotlib 
import matplotlib.pyplot as plt
import math, random, os, pickle, mmh3, time
import numpy as np
from datetime import datetime
import settings, PerfectCounter, Buckets, NiceBuckets, SEAD_stat, SEAD_dyn, F2P_li, F2P_si, Morris, CEDAR
from settings import warning, error, VERBOSE_RES, VERBOSE_PCL, VERBOSE_LOG, VERBOSE_DETAILED_LOG, VERBOSE_LOG_END_SIM, calcPostSimStat
from   tictoc import tic, toc # my modules for measuring and print-out the simulation time.
from printf import printf, printarFp
from SingleCntrSimulator import getFxpCntrMaxVal, genCntrMasterFxp
from CountMinSketch import CountMinSketch

class SpaceSaving (CountMinSketch):

    # given the flowId and the row, return the hash, namely, the corresponding counter in that row  
    hashOfFlow = lambda self, flowId, row : mmh3.hash(str(flowId), seed=self.seed + row) % self.width

    # given the flowId, return the list of cntrs hashed to this flow Id.   
    hashedCntrsOfFlow = lambda self, flowId : [self.mat2aridx(row, mmh3.hash(str(flowId), seed=self.seed + row) % self.width) for row in range(self.depth)] 

    # given the row and col. in a matrix, return the corresponding index if the mat is flattened into a 1D array.
    mat2aridx  = lambda self, row, col       : self.width*row + col 

    # Generate a string that details the parameters' values.
    genSettingsStr = lambda self : f'ss_{self.traceFileName}_{self.mode}_bit{self.cntrSize}'
    
    def __init__(self,
        mode            = 'PerfectCounter', # the counter mode (e.g., SEC, AEE, realCounter).
        cntrSize        = 2, # num of bits in each counter
        verbose         = [], # The chosen verbose options, detailed in settings.py, determine the output (e.g., to a .pcl, .res or .log file).
        numCntrs        = 1, # number of counters -- actually, the cache's size
        seed            = settings.SEED,
    ):
        
        """
        """
        self.cntrSize   = cntrSize
        self.mode, self.seed = mode, seed
        random.seed (self.seed)
        self.numCntrs   = numCntrs
        self.verbose = verbose
        self.genOutputDirectories ()


    def queryFlow(self, flow):
        """
        """
        res = math.inf       
        for row in range(self.depth):
            res = min (res, self.cntrMaster.queryCntr(cntrIdx=self.mat2aridx (row=row, col=self.hashOfFlow (flowId=flowId, row=row)) , factor=1, mult=False))
        return res
   
    def openOutputFiles (self) -> None:
        """
        Open the output files (.res, .log, .pcl), as defined by the verbose level requested.
        """      
        if VERBOSE_PCL in self.verbose:
            self.pclOutputFile = open(f'../res/pcl_files/ss_M{self.numCntrs}_{settings.getMachineStr()}.pcl', 'ab+')

        if (VERBOSE_RES in self.verbose):
            self.resFile = open (f'../res/ss_M{self.numCntrs}__{settings.getMachineStr()}.res', 'a+')
            
        if (settings.VERBOSE_FULL_RES in self.verbose):
            self.fullResFile = open (f'../res/ss_M{self.numCntrs}_{settings.getMachineStr()}_full.res', 'a+')

        if VERBOSE_LOG in self.verbose or VERBOSE_DETAILED_LOG in self.verbose:
            self.logFile = open (f'../res/log_files/{self.genSettingsStr()}.log', 'w')
            
    def printSimMsg (self, str):
        """
        Print-screen an info msg about the parameters and hours of the simulation starting to run. 
        """             
        print ('{} running sim at t={}. trace={}, numOfExps={}, mode={}, cntrSize={}, numCntrs={}' .format (
                        str, datetime.now().strftime('%H:%M:%S'), self.traceFileName, self.numOfExps, self.mode, self.cntrSize, self.numCntrs))

    def runSimFromTrace (self):
        """
        Run a simulation where the input is taken from self.traceFileName.
        """

        if self.numFlows==None:
            error ('In CountMinSketch.runSimFromTrace(). Sorry, dynamically calculating the flowNum is not supported yet.')
        self.genCntrMaster ()
        if (VERBOSE_LOG in self.verbose) or (VERBOSE_LOG_END_SIM in self.verbose):
            infoStr = '{}_{}' .format (self.genSettingsStr(), self.cntrMaster.genSettingsStr())
            self.logFile = open (f'../res/log_files/{infoStr}.log', 'w')
            self.cntrMaster.setLogFile(self.logFile)

        relativePathToInputFile = settings.getRelativePathToTraceFile (self.traceFileName)
        settings.checkIfInputFileExists (relativePathToInputFile)
        for self.expNum in range (self.numOfExps):
            self.seed = self.expNum+1 
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
                    printf (self.logFile, 'incNum={}, hashes={}, estimatedVal={:.0f} realVal={:.0f} \n' .format(self.incNum, self.hashedCntrsOfFlow(flowId), flowEstimatedVal, flowRealVal[flowId])) 
                if VERBOSE_DETAILED_LOG in self.verbose and self.incNum>10000: #$$$
                    printf (self.logFile, 'incNum={}, realVal={}, estimated={:.1e}, sqAbsEr={:.1e}, sqRelEr={:.1e}, sumAbsSqEr={:.1e}, sumRelSqEr={:.1e}\n' .format (self.incNum, flowRealVal[flowId], flowEstimatedVal, sqEr, sqEr/(flowRealVal[flowId])**2, self.sumSqAbsEr[self.expNum], self.sumSqRelEr[self.expNum]))
                    # printf (self.logFile, f'incNum={}, realVal={}, estimated={:.1e}, sqAbsEr={:.1e}, sqRelEr={:.1e}, sumAbsSqEr={:.1e}\n' .format (self.incNum, flowRealVal[flowId], flowEstimatedVal, sqEr, sqEr/(flowRealVal[flowId])**2, self.sumSqAbsEr[self.expNum]))
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
                # flowId = mmh3.hash(str(flowId)) % self.numFlows
                flowRealVal[flowId]     += 1
                flowEstimatedVal   = self.incNQueryFlow (flowId=flowId)
                sqEr = (flowRealVal[flowId] - flowEstimatedVal)**2
                self.sumSqAbsEr[self.expNum] += sqEr                
                self.sumSqRelEr[self.expNum] += sqEr/(flowRealVal[flowId])**2                
                if VERBOSE_LOG in self.verbose:
                    self.cntrMaster.printAllCntrs (self.logFile)
                    printf (self.logFile, 'incNum={}, hashes={}, estimatedVal={:.0f} realVal={:.0f} \n' .format(self.incNum, self.hashedCntrsOfFlow(flowId), flowEstimatedVal, flowRealVal[flowId])) 
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
                    verbose         = self.verbose
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
        dict['cntrSize']    = self.cntrSize
        dict['depth']       = self.depth
        dict['width']       = self.width
        dict['numFlows']    = self.numFlows
        dict['seed']        = self.seed
        return dict
    
def runCMS (mode, 
    cntrSize    = 8,
    runShortSim = True,
    maxValBy    = 'f2p_li_h2',
    maxNumIncs  = float ('inf'),
    width       = 2**10,
    depth       = 4,
):
    """
    """   
    traceFileName   = 'Caida1' 
    numFlows = 1276112 # 13,182,023 incs
    
    if runShortSim:
        width, depth            = 2, 2
        numFlows                = numFlows
        numCntrsPerBkt          = 2
        maxNumIncs              = 20 #4000 #(width * depth * cntrSize**3)/2
        numOfExps               = 2
        numEpsilonStepsIceBkts  = 5 
        numEpsilonStepsInRegBkt = 2
        numEpsilonStepsInXlBkt  = 5
        verbose                 = [VERBOSE_RES] # VERBOSE_LOG, VERBOSE_LOG_END_SIM, VERBOSE_LOG, settings.VERBOSE_DETAILS
    else:
        width, depth            = width, depth
        numFlows                = numFlows
        numCntrsPerBkt          = 1 #16
        maxNumIncs              = maxNumIncs   
        numOfExps               = 10 #$$$ #100 
        numEpsilonStepsIceBkts  = 6 
        numEpsilonStepsInRegBkt = 5
        numEpsilonStepsInXlBkt  = 7
        verbose                 = [VERBOSE_RES, VERBOSE_PCL] #$$$ [VERBOSE_RES, VERBOSE_PCL] # VERBOSE_LOG_END_SIM,  VERBOSE_RES, settings.VERBOSE_FULL_RES, VERBOSE_PCL] # VERBOSE_LOG, VERBOSE_RES, VERBOSE_PCL, settings.VERBOSE_DETAILS
    
    cms = CountMinSketch (
        width       = width, 
        depth       = depth, 
        cntrSize    = cntrSize, 
        numFlows    = numFlows, 
        verbose     = verbose,
        mode        = mode,
        maxValBy    = maxValBy,  
        numCntrsPerBkt          = numCntrsPerBkt,
        numEpsilonStepsIceBkts  = numEpsilonStepsIceBkts, 
        numEpsilonStepsInRegBkt = numEpsilonStepsInRegBkt, 
        numEpsilonStepsInXlBkt  = numEpsilonStepsInXlBkt,
        )
    cms.hyperSize = 2
    cms.sim (
        numOfExps      = numOfExps, 
        maxNumIncs     = maxNumIncs, 
        traceFileName  = traceFileName
        )
    
if __name__ == '__main__':
    try:
        cntrSize = 8
        for width in [2**i for i in range (10, 19)]:
            for mode in ['SEAD_dyn']:    
            # for mode in ['F2P_li_h2']:    
            # for mode in ['F3P_li_h3']:    
            # for mode in ['F2P_lli', 'CEDAR', 'Morris']:    
                runCMS (
                    mode        = mode, 
                    cntrSize    = cntrSize, 
                    runShortSim = False,
                    maxValBy    = 'Caida1',
                    width       = width
                )
    except KeyboardInterrupt:
        print('Keyboard interrupt.')
