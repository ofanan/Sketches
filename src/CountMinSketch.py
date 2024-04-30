import matplotlib 
import matplotlib.pyplot as plt
import math, random, os, pickle, mmh3, time
import numpy as np
from datetime import datetime
import settings, PerfectCounter, Buckets, NiceBuckets, SEAD_stat, SEAD_dyn, F2P_li, F2P_si, Morris, CEDAR
from settings import warning, error, VERBOSE_RES, VERBOSE_PCL, VERBOSE_LOG, VERBOSE_DETAILED_LOG, VERBOSE_LOG_END_SIM, calcPostSimStat
from   tictoc import tic, toc # my modules for measuring and print-out the simulation time.
from printf import printf, printarFp

class CountMinSketch:

    # given the flowId and the row, return the hash, namely, the corresponding counter in that row  
    hashOfFlow = lambda self, flowId, row : mmh3.hash(str(flowId), seed=self.seed + row) % self.width

    # given the flowId, return the list of cntrs hashed to this flow Id.   
    hashedCntrsOfFlow = lambda self, flowId : [self.mat2aridx(row, mmh3.hash(str(flowId), seed=self.seed + row) % self.width) for row in range(self.depth)] 

    # given the row and col. in a matrix, return the corresponding index if the mat is flattened into a 1D array.
    mat2aridx  = lambda self, row, col       : self.width*row + col 

    # Generate a string that details the parameters' values.
    genSettingsStr = lambda self : f'cms_{self.traceFileName}_{self.mode}_d{self.depth}_w{self.width}_f{self.numFlows}_bit{self.cntrSize}'
    
    def __init__(self,
                 width          = 2, # the number of counters per row.
                 depth          = 2, # the number of rows or hash functions.
                 numCntrsPerBkt = 2,
                 numFlows       = 10, # the total number of flows to be estimated.
                 mode           = 'PerfectCounter', # the counter mode (e.g., SEC, AEE, realCounter).
                 cntrSize       = 2, # num of bits in each counter
                 verbose        = [], # The chosen verbose options, detailed in settings.py, determine the output (e.g., to a .pcl, .res or .log file).
                 seed           = settings.SEED,
                 numEpsilonStepsIceBkts  = 6, # number of "epsilon" steps in Ice Buckets.
                 numEpsilonStepsInRegBkt = 5, # number of "epsilon" steps in regular buckets in NiceBuckets.
                 numEpsilonStepsInXlBkt  = 6  # number of "epsilon" steps in the XL buckets in NiceBuckets.
                 ):
        
        """
        """
        random.seed (settings.SEED)
        self.numCntrsPerBkt = int(numCntrsPerBkt)
        if depth<1 or width<1 or cntrSize<1:
            error (f'CountMinSketch was called with depth={depth}, width={width}, cntrSize={cntrSize}. All these parameters should be at least 1.')
        if depth<2 or width<2:
            print (f'Note: CountMinSketch was called with depth={depth} and width={width}.')            
        self.cntrSize, self.width, self.depth, self.numFlows = cntrSize, width, depth, numFlows
        self.mode, self.seed = mode, seed
        self.numEpsilonStepsInRegBkt    = numEpsilonStepsInRegBkt
        self.numEpsilonStepsInXlBkt     = numEpsilonStepsInXlBkt
        self.numEpsilonStepsIceBkts     = numEpsilonStepsIceBkts
        self.numCntrs   = self.width * self.depth
        
        numBucketsFP = self.numCntrs / self.numCntrsPerBkt
        self.numBuckets = int(numBucketsFP)
        if numBucketsFP != self.numBuckets:
            error (f'You chose numCntrs={self.numCntrs}, numCntrsPerBkt={self.numCntrsPerBkt}. However, numCntrs should be divisible by numCntrsPerBkt')           
        if self.numBuckets < 1:
            error (f'numOfBkts={self.numOfBkts}')
        self.conf       = settings.getConfByCntrSize (cntrSize=self.cntrSize)
        self.verbose = verbose
        self.genOutputDirectories ()

    def genCntrMaster (self):
        """
        Generate self.cntrMaster according to the mode requested
        self.cntrMaster is the entity that manages the counters - including incrementing and querying counters.
        Documentation about the various CntrMaster's types is found in the corresponding .py files. 
        """
        myF2P_cntrMaster = F2P_li.CntrMaster (cntrSize=self.cntrSize, hyperSize=self.hyperSize)
        cntrMaxVal = myF2P_cntrMaster.getCntrMaxVal ()
        # cntrMaxVal      = settings.getCntrMaxValByCntrSize (self.cntrSize),
        if self.mode=='PerfectCounter':
            self.cntrMaster = PerfectCounter.CntrMaster (
                cntrSize    = self.cntrSize, 
                numCntrs    = self.numCntrs, 
                verbose     = self.verbose)
        elif self.mode.startswith('SEAD_stat'):
            expSize = int(self.mode.split('_e')[1])
            self.cntrMaster = SEAD_stat.CntrMaster (
                cntrSize        = self.cntrSize, 
                numCntrs        = self.numCntrs, 
                expSize         = expSize,
                verbose         = self.verbose)
        elif self.mode=='SEAD_dyn':
            self.cntrMaster = SEAD_dyn.CntrMaster (
                cntrSize        = self.cntrSize, 
                numCntrs        = self.numCntrs, 
                verbose         = self.verbose)
        elif self.mode.startswith('F2P_li_'):
            hyperSize = int(self.mode.split('_h')[1])
            self.cntrMaster = F2P_li.CntrMaster (
                cntrSize        = self.cntrSize, 
                numCntrs        = self.numCntrs, 
                hyperSize       = hyperSize,
                verbose         = self.verbose)
        elif self.mode.startswith('F2P_si_'):
            hyperSize = int(self.mode.split('_h')[1])
            self.cntrMaster = F2P_si.CntrMaster (
                cntrSize        = self.cntrSize, 
                numCntrs        = self.numCntrs, 
                hyperSize       = hyperSize,
                verbose         = self.verbose)
        elif self.mode=='Morris':
            self.cntrMaster = Morris.CntrMaster (
                cntrSize        = self.cntrSize, 
                numCntrs        = self.numCntrs,
                cntrMaxVal      = cntrMaxVal,
                verbose         = self.verbose)
        elif self.mode=='CEDAR': 
            self.cntrMaster = CEDAR.CntrMaster (
                cntrSize        = self.cntrSize, 
                numCntrs        = self.numCntrs, 
                cntrMaxVal      = cntrMaxVal,
                verbose         = self.verbose)
        elif self.mode=='IceBuckets':
            self.cntrMaster = Buckets.Buckets (
                cntrSize        = self.cntrSize, 
                numCntrs        = self.numCntrs, 
                numCntrsPerBkt  = self.numCntrsPerBkt, 
                mode            = 'ICE',
                numEpsilonSteps = self.numEpsilonStepsIceBkts,
                verbose         = self.verbose)
        elif self.mode=='NiceBuckets':
            self.cntrMaster = NiceBuckets.CntrMaster (
                cntrSize                = self.cntrSize, 
                numCntrs                = self.numCntrs, 
                numCntrsPerRegBkt       = self.numCntrsPerBkt,
                numCntrsPerXlBkt        = self.numCntrsPerBkt,
                numEpsilonStepsInRegBkt = self.numEpsilonStepsInRegBkt,
                numEpsilonStepsInXlBkt  = self.numEpsilonStepsInXlBkt,
                numXlBkts               = self.depth,
                verbose                 = self.verbose)
        elif self.mode=='SecBuckets':
             self.cntrMaster = Buckets.Buckets (
                cntrSize        = self.cntrSize, 
                numCntrs        = self.numCntrs, 
                numCntrsPerBkt  = self.numCntrsPerBkt, 
                mode            = 'SEC', 
                verbose         = self.verbose)
        elif self.mode=='F2pBuckets':
            self.cntrMaster = Buckets.Buckets (
                cntrSize        = self.cntrSize, 
                numCntrs        = self.numCntrs, 
                numCntrsPerBkt  = self.numCntrsPerBkt, 
                mode            = 'F2P',
                verbose         = self.verbose)
        elif self.mode=='MecBuckets':
            self.cntrMaster = Buckets.Buckets (
                cntrSize        = self.cntrSize, 
                numCntrs        = self.numCntrs, 
                numCntrsPerBkt  = self.numCntrsPerBkt, 
                mode            = 'MEC',
                verbose         = self.verbose)
        else:
            warning (f'Sorry, the mode {self.mode} that you requested is not supported')

    
    def genOutputDirectories (self):
        """
        Generate and open the directories for output (results, logs, etc.):
        - Generate directories for the output files if not exist
        - Verify the verbose level requested.
        """      
        if settings.VERBOSE_DETAILED_RES in self.verbose or settings.VERBOSE_FULL_RES in self.verbose:
            self.verbose.append (VERBOSE_RES)
        # if not (VERBOSE_PCL in self.verbose):
        #     print ('Note: verbose does not include .pcl')  
        
        pwdStr = os.getcwd()
        if (pwdStr.find ('itamarc')>-1): # the string 'HPC' appears in the path only in HPC runs
            self.machineStr  = 'HPC' # indicates that this sim runs on my PC
        else:
            self.machineStr  = 'PC' # indicates that this sim runs on an HPC       
        if not (os.path.exists('../res')):
            os.makedirs ('../res')
        if not (os.path.exists('../res/log_files')):
            os.makedirs ('../res/log_files')
        if not (os.path.exists('../res/pcl_files')):
            os.makedirs ('../res/pcl_files')

    def incNQueryFlow(self, flowId, mult=False, factor=1):
        """
        Update the value for a single flow. 
        Return the updated estimated value for this flow.
        - Update the corresponding counters.
        - Return the minimum of the corresponding counters.
        """
        flowValAfterInc = math.inf
        for row in range(self.depth):
            flowValAfterInc = min (flowValAfterInc, self.cntrMaster.incCntrBy1GetVal(cntrIdx=self.mat2aridx (row=row, col=self.hashOfFlow (flowId=flowId, row=row))))
        return flowValAfterInc

    def queryFlow(self, flow):
        """
        """
        res = math.inf       
        for row in range(self.depth):
            res = min (res, self.cntrMaster.queryCntr(cntrIdx=self.mat2aridx (row=row, col=self.hashOfFlow (flowId=flowId, row=row)) , factor=1, mult=False))
        return res
   
    def dumpDictToPcl (self, dict):
        """
        Dump a single dict of data into pclOutputFile
        """
        if (VERBOSE_PCL in self.verbose):
            pickle.dump(dict, self.pclOutputFile) 
    
    def writeDictToResFile (self, dict):
        """
        Write a single dict of data into resOutputFile
        """
        if (VERBOSE_RES in self.verbose):
            printf (self.resFile, f'{dict}\n\n') 
        if (settings.VERBOSE_FULL_RES in self.verbose):
            printf (self.fullResFile, f'{dict}\n\n') 
    

    def openOutputFiles (self) -> None:
        """
        Open the output files (.res, .log, .pcl), as defined by the verbose level requested.
        """      
        if VERBOSE_PCL in self.verbose:
            self.pclOutputFile = open(f'../res/pcl_files/cms_{settings.getMachineStr()}.pcl', 'ab+')

        if (VERBOSE_RES in self.verbose):
            self.resFile = open (f'../res/cms_{settings.getMachineStr()}.res', 'a+')
            
        if (settings.VERBOSE_FULL_RES in self.verbose):
            self.fullResFile = open (f'../res/cms_full.res', 'a+')

    def printSimMsg (self, str):
        """
        Print-screen an info msg about the parameters and hours of the simulation starting to run. 
        """             
        print ('{} running sim at t={}. trace={}, numOfExps={}, mode={}, cntrSize={}, depth={}, width={}, numFlows={}' .format (
                        str, datetime.now().strftime('%H:%M:%S'), self.traceFileName, self.numOfExps, self.mode, self.cntrSize, self.depth, self.width, self.numFlows))

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
        
        self.openOutputFiles ()
        self.maxNumIncs, self.numOfExps, self.traceFileName = maxNumIncs, numOfExps, traceFileName
        self.sumSqAbsEr  = [0] * self.numOfExps # self.sumSqAbsEr[j] will hold the sum of the square absolute errors collected at experiment j. 
        self.sumSqRelEr  = [0] * self.numOfExps # self.sumSqRelEr[j] will hold the sum of the square relative errors collected at experiment j.        
        self.printSimMsg ('Started')
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

                
    def writeProgress (self, infoStr=None):
        """
        If the verbose requires that, report the progress to self.logFile
        """ 
        if not (settings.VERBOSE_PROGRESS in self.verbose):
            return
        if infoStr==None:
            printf (self.logFile, f'starting experiment{self.expNum}\n')
        else:
            printf (self.logFile, f'{infoStr}\n')
    
    def fillStatDictsFields (self, dict) -> dict:
        """
        Add to the given dict some fields detailing the sim settings. Return the full dict.
        """
        dict['numOfExps']   = self.expNum+1# The count of the experiments started in 0
        dict['numIncs']     = self.incNum
        dict['mode']        = self.mode
        dict['cntrSize']    = self.mode
        dict['depth']       = self.depth
        dict['width']       = self.width
        dict['numFlows']    = self.numFlows
        dict['seed']        = self.seed
        return dict
    
def runCMS (mode, 
            cntrSize    = 8,
            runShortSim = True,
            maxNumIncs  = float ('inf')):
    """
    """   
    traceFileName   = 'Caida1' 
    numFlows = 1276112
    
    if runShortSim:
        width, depth            = 2, 2
        numFlows                = numFlows
        numCntrsPerBkt          = 2
        maxNumIncs              = 20 #4000 #(width * depth * cntrSize**3)/2
        numOfExps               = 2
        numEpsilonStepsIceBkts  = 5 
        numEpsilonStepsInRegBkt = 2
        numEpsilonStepsInXlBkt  = 5
        verbose                 = [VERBOSE_LOG_END_SIM] # VERBOSE_LOG, VERBOSE_LOG_END_SIM, VERBOSE_LOG, settings.VERBOSE_DETAILS
    else:
        width, depth            = 32, 2 #$$$ 1024, 4
        numFlows                = numFlows
        numCntrsPerBkt          = 1 #16
        maxNumIncs              = 20 #maxNumIncs   
        numOfExps               = 2
        numEpsilonStepsIceBkts  = 6 
        numEpsilonStepsInRegBkt = 5
        numEpsilonStepsInXlBkt  = 7
        verbose                 = [VERBOSE_LOG_END_SIM] # VERBOSE_LOG_END_SIM,  VERBOSE_RES, settings.VERBOSE_FULL_RES, VERBOSE_PCL] # VERBOSE_LOG, VERBOSE_RES, VERBOSE_PCL, settings.VERBOSE_DETAILS
    
    cms = CountMinSketch (
        width       = width, 
        depth       = depth, 
        cntrSize    = cntrSize, 
        numFlows    = numFlows, 
        verbose     = verbose,
        mode        = mode, 
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
        for cntrSize in [6]:# , 10, 12]: # 14, 16]:
            for mode in ['F2P_si_h2']: #, 'PerfectCounter', 'CEDAR', 'F2P_li_h1', 'SEAD_dyn']:   
                runCMS (mode=mode, cntrSize=cntrSize, runShortSim=False)
    except KeyboardInterrupt:
        print('Keyboard interrupt.')
