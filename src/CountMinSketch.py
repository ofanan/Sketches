from ttictoc import tic,toc
import matplotlib 
import matplotlib.pyplot as plt
import math, random, os, pickle, mmh3, time
import numpy as np

from datetime import datetime

import settings, PerfectCounter, Buckets, NiceBuckets, SEAD_stat, SEAD_dyn, F2P_si, Morris, CEDAR, CEDAR_ds, AEE_ds
from settings import * 
from printf import printf, printarFp
from SingleCntrSimulator import getCntrMaxValFromFxpStr, genCntrMasterFxp
np.set_printoptions(precision=1)

class CountMinSketch:

    # given the row and col. in a matrix, return the corresponding index if the mat is flattened into a 1D array.
    mat2aridx  = lambda self, row, col       : self.width*row + col 

    # Generate a string that details the parameters' values.
    genSettingsStr = lambda self : f'cms_{self.traceName}_{self.mode}_d{self.depth}_w{self.width}_bit{self.cntrSize}'
    
    def __init__(self,
            width           = 2, # the number of counters per row.
            depth           = 2, # the number of rows or hash functions.
            numCntrsPerBkt  = 2,
            mode            = 'PerfectCounter', # the counter mode (e.g., SEC, AEE, realCounter).
            cntrSize        = 8, # num of bits in each counter
            verbose         = [], # The chosen verbose options, detailed in settings.py, determine the output (e.g., to a .pcl, .res or .log file).
            seed            = settings.SEED,
            maxValBy        = None, # How to calculate the maximum value (for SEAD/CEDAR).   
            maxNumIncs      = INF_INT, # maximum # of increments (pkts in the trace), after which the simulation will be stopped. 
            numOfExps       = 1,  # number of repeated experiments. Relevant only for randomly-generated traces.
            traceName       = 'Rand',
            numEpsilonStepsIceBkts  = 6, # number of "epsilon" steps in Ice Buckets.
            numEpsilonStepsInRegBkt = 5, # number of "epsilon" steps in regular buckets in NiceBuckets.
            numEpsilonStepsInXlBkt  = 6,  # number of "epsilon" steps in the XL buckets in NiceBuckets.
        ):
        
        """
        """
        self.mode, self.seed = mode, seed
        self.numCntrsPerBkt  = int(numCntrsPerBkt)
        self.traceName       = traceName
        self.maxValBy        = maxValBy
        self.dwnSmpl         = self.mode.endswith('_ds')
        if depth<1 or width<1 or cntrSize<1:
            error (f'CountMinSketch__init() was called with depth={depth}, width={width}, cntrSize={cntrSize}. All these parameters should be at least 1.')
        if depth<2 or width<2:
            warning (f'CountMinSketch__init() was called with depth={depth} and width={width}.')            
        self.cntrSize, self.width, self.depth = cntrSize, width, depth
        self.maxNumIncs, self.numOfExps, = maxNumIncs, numOfExps
        if self.traceName=='Rand':
            self.cntrMaxVal = self.maxNumIncs
        else:
            if self.maxValBy==None: # By default, the maximal counter's value is the trace length 
                self.cntrMaxVal = getTraceLen(self.traceName)
            elif maxValBy=='int':
                self.cntrMaxVal = 2**self.cntrSize - 1
            elif maxValBy.startswith ('F2P') or maxValBy.startswith ('F3P'):
                self.cntrMaxVal = getCntrMaxValFromFxpStr (
                    cntrSize        = self.cntrSize, 
                    fxpSettingStr   = self.maxValBy)
            else:
                error (f'In CountMinSketch.init(). the chosen maxValBy {maxValBy} is not supported.')
        random.seed (self.seed)
        self.numEpsilonStepsInRegBkt    = numEpsilonStepsInRegBkt
        self.numEpsilonStepsInXlBkt     = numEpsilonStepsInXlBkt
        self.numEpsilonStepsIceBkts     = numEpsilonStepsIceBkts
        self.numCntrs   = self.width * self.depth
        
        numBucketsFP = self.numCntrs / self.numCntrsPerBkt
        self.numBuckets = int(numBucketsFP)
        if numBucketsFP != self.numBuckets:
            error (f'CountMinSketch__init() was called with numCntrs={self.numCntrs}, numCntrsPerBkt={self.numCntrsPerBkt}. However, numCntrs should be divisible by numCntrsPerBkt')           
        if self.numBuckets < 1:
            error (f'CountMinSketch.__init() was called with numOfBkts={self.numOfBkts}')
        self.verbose = verbose
        self.genOutputDirectories ()

    def genCntrMaster (self):
        """
        Generate self.cntrMaster according to the mode requested
        self.cntrMaster is the entity that manages the counters - including incrementing and querying counters.
        Documentation about the various CntrMaster's types is found in the corresponding .py files. 
        """
        if self.mode=='PerfectCounter':
            self.cntrMaster = PerfectCounter.CntrMaster (
                cntrSize    = self.cntrSize, 
                numCntrs    = self.numCntrs, 
                verbose     = self.verbose)
        elif self.mode.startswith('SEAD_stat'):
            expSize = getSeadStatExpSize (mode)
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
        elif self.mode.startswith('F2P') or self.mode.startswith('F3P'):
            self.cntrMaster     = genCntrMasterFxp (
                cntrSize        = self.cntrSize,
                numCntrs        = self.numCntrs,
                fxpSettingStr   = self.mode, 
                verbose         = self.verbose
            )
        elif self.mode=='Morris':
            self.cntrMaster = Morris.CntrMaster (
                cntrSize        = self.cntrSize, 
                numCntrs        = self.numCntrs,
                cntrMaxVal      = self.cntrMaxVal,
                verbose         = self.verbose)
        elif self.mode=='CEDAR_ds': 
            self.cntrMaster = CEDAR_ds.CntrMaster (
                cntrSize        = self.cntrSize, 
                numCntrs        = self.numCntrs, 
                cntrMaxVal      = self.cntrMaxVal,
                verbose         = self.verbose)
        elif self.mode=='CEDAR': 
            self.cntrMaster = CEDAR.CntrMaster (
                cntrSize        = self.cntrSize, 
                numCntrs        = self.numCntrs, 
                cntrMaxVal      = self.cntrMaxVal,
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
        elif self.mode=='AEE_ds':
            self.cntrMaster = AEE_ds.CntrMaster (
                cntrSize        = self.cntrSize, 
                numCntrs        = self.numCntrs,
                cntrMaxVal      = self.cntrMaxVal,
                verbose         = self.verbose)
        else:
            error (f'In CountMinSketch.genCntrMaster(). Sorry, the mode {self.mode} that you requested is not supported')
        
    
    def genOutputDirectories (self):
        """
        Generate and open the directories for output (results, logs, etc.):
        - Generate directories for the output files if not exist
        - Verify the verbose level requested.
        """      
        if VERBOSE_DETAILED_RES in self.verbose or VERBOSE_FULL_RES in self.verbose:
            self.verbose.append (VERBOSE_RES)
        if not (VERBOSE_PCL in self.verbose):
            print ('Note: verbose does not include .pcl')  
        
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

    def queryFlow (
            self, 
            flowId, 
        ):
        """
        Returns the estimated value for this flow, namely, the minimum of the corresponding counters.
        """
        val = math.inf
        for row in range(self.depth):
            val = min (
                val, 
                self.cntrMaster.cntr2num (
                    self.cntrMaster.cntrs[self.mat2aridx (row=row, col=self.hashOfFlow (flowId=flowId, row=row))]
                )
            )
        return val

    def incNQueryFlow (
            self, 
            hashes : np.array, # hashes[i] is the index to access in the i-th row  
            mult   : bool = False,   # When True, perform multiplicative increment 
            factor : int = 1        # the weight to increment
        ) -> float:
        """
        Update the value for a single flow. 
        Return the updated estimated value for this flow.
        - Update the corresponding counters.
        - Return the minimum of the corresponding counters.
        """
        if mult or factor!=1:
            error ('In CountMinSketh.py(). Sorry, multiplicative increment or factor!=1 are currently not supported.')
        flowValAfterInc = math.inf
        for row in range(self.depth):
            flowValAfterInc = min (
                flowValAfterInc, 
                self.cntrMaster.incCntrBy1GetVal(cntrIdx=self.mat2aridx (row=row, col=hashes[row])) 
            )
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
        if (VERBOSE_FULL_RES in self.verbose):
            printf (self.fullResFile, f'{dict}\n\n') 
    

    def openOutputFiles (self) -> None:
        """
        Open the output files (.res, .log, .pcl), as defined by the verbose level requested.
        """      
        maxValByStr = getMaxValByStr (self.maxValBy)          
        if VERBOSE_PCL in self.verbose:
            self.pclOutputFile = open(f'../res/pcl_files/cms_{self.traceName}_{getMachineStr()}_by_{maxValByStr}.pcl', 'ab+')

        if (VERBOSE_RES in self.verbose):
            self.resFile = open (f'../res/cms_{self.traceName}_{getMachineStr()}.res', 'a+')
            
        if (VERBOSE_FULL_RES in self.verbose):
            self.fullResFile = open (f'../res/cms_full.res', 'a+')

        self.logFile =  None # default

        if VERBOSE_LOG in self.verbose or \
           VERBOSE_LOG_SHORT in self.verbose or \
           VERBOSE_PROGRESS in self.verbose or \
           VERBOSE_LOG_END_SIM in self.verbose or \
           VERBOSE_LOG_DWN_SMPL in self.verbose:
           self.logFile = open (f'../res/log_files/{self.genSettingsStr()}_by_{maxValByStr}.log', 'w')
            
    def printSimMsg (self, str):
        """
        Print-screen an info msg about the parameters and hours of the simulation starting to run. 
        """             
        print ('{} running cms at t={}. trace={}, maxValBy={}, numOfExps={}, mode={}, cntrSize={}, depth={}, width={}, numFlows={}, verbose={}' .format (
                        str, datetime.now().strftime('%H:%M:%S'), self.traceName, self.maxValBy, self.numOfExps, self.mode, self.cntrSize, self.depth, self.width, self.numFlows, self.verbose))

    def logEndSim (self):
        """
        If VERBOSE_LOG_END_SIM is in the chosen verbose, output to a log file data 
        about the counters' values at the end of the sim. 
        """
        if VERBOSE_LOG_END_SIM in self.verbose:
            printf (self.logFile, '\n// At the end of sim:\n')
            self.cntrMaster.printCntrsStat (
                self.logFile, 
            )
    
    def rmvVerboseLogs (self):
        """
        Rmv all the "log" verboses from self.verbose. To be used after the first experiment, as no need to log more than a single exp.
        """
        
        for verbose in [VERBOSE_LOG_SHORT, VERBOSE_LOG, VERBOSE_DETAILED_LOG, VERBOSE_LOG_DWN_SMPL]:
            if verbose in self.verbose:
                self.verbose.remove(verbose)
                self.cntrMaster.rmvVerbose(verbose)
    
    def calcTraceHashes (self) ->np.array:
        """
        returns a 2D array that contains, at each row, all the (depth) hashes of the flowId at that row. 
        """
        if self.depth*self.width < 2**16:
            traceHashes = np.zeros ([self.maxNumIncs, self.depth], dtype='uint16')
        else:
            traceHashes = np.zeros ([self.maxNumIncs, self.depth], dtype='uint32')            
        for depth in range(self.depth):
            traceHashes[:,depth] = (self.traceKeys + depth) % self.width
        return traceHashes 
        
    def sim (
        self, 
        numFlows : int = 4, # number of flows when running a synthetic sim. When reading from a trace, the # of flows is found according to the trace. 
        ):
        """
        Simulate the count min sketch
        """
        
        self.sumSqAbsEr  = np.zeros (self.numOfExps) # self.sumSqAbsEr[j] will hold the sum of the square absolute errors collected at experiment j. 
        self.sumSqRelEr  = np.zeros (self.numOfExps) # self.sumSqRelEr[j] will hold the sum of the square relative errors collected at experiment j.        
        self.openOutputFiles ()

        if self.traceName=='Rand': # run synthetic, randomized trace
            rng = np.random.default_rng()
            self.numFlows     = numFlows
            self.traceFlowIds = rng.integers (self.numFlows, size=self.maxNumIncs, dtype='uint32')
            self.traceKeys    = self.traceFlowIds # for rand sim, the trace keys are equal to the flow Ids. 
        else:
            relativePathToInputFile = getRelativePathToTraceFile (f'{getTraceFullName(self.traceName)}_flowIds.pcl')
            checkIfInputFileExists (relativePathToInputFile, exitError=True)
            with open (relativePathToInputFile, 'rb') as file:
                self.traceFlowIds = np.array (pickle.load(file))
            file.close ()
            relativePathToInputFile = getRelativePathToTraceFile (f'{getTraceFullName(self.traceName)}_flowId2key.pcl')
            checkIfInputFileExists (relativePathToInputFile, exitError=True)
            with open (relativePathToInputFile, 'rb') as file:
                flowId2key     = np.array (pickle.load(file))
            file.close ()
            self.numFlows     = flowId2key.shape[0]  
            self.traceKeys = np.array([flowId2key[flowId] for flowId in self.traceFlowIds])
            self.maxNumIncs = min (self.maxNumIncs, self.traceKeys.shape[0])
            self.traceKeys = self.traceKeys[:self.maxNumIncs] # If the trace is longer than the requested # of increments, trunc it to the desired size to save space.
        traceHashes = self.calcTraceHashes ()
        self.printSimMsg ('Started')
        tic ()
        for self.expNum in range (self.numOfExps):
            self.seed = self.expNum+1
            random.seed (self.seed) 
            self.genCntrMaster () # Generate a fresh, empty CntrMaster, for each experiment
            self.cntrMaster.setLogFile(self.logFile)
            flowRealVal = np.zeros(self.numFlows)
            self.writeProgress () # log the beginning of the experiment; used to track the progress of long runs.

            for self.incNum in range(self.maxNumIncs):
                flowId = self.traceFlowIds[self.incNum]            
                flowRealVal[flowId]     += 1
                flowEstimatedVal = self.incNQueryFlow (hashes=traceHashes[self.incNum])
                sqEr = (flowRealVal[flowId] - flowEstimatedVal)**2
                self.sumSqAbsEr[self.expNum] += sqEr    
                self.sumSqRelEr[self.expNum] += sqEr/(flowRealVal[flowId])**2                
                if VERBOSE_LOG_SHORT in self.verbose: 
                    printf (self.logFile, '\nincNum={}, hashes={}, estimatedVal={:.0f} realVal={:.0f} \n' .format(
                        self.incNum, traceHashes[self.incNum], flowEstimatedVal, flowRealVal[flowId]))
                    self.cntrMaster.printAllCntrs (self.logFile, printAlsoVec=False)
                elif VERBOSE_LOG in self.verbose: 
                    printf (self.logFile, '\nincNum={}, hashes={}, estimatedVal={:.0f} realVal={:.0f} \n' .format(
                        self.incNum, traceHashes[self.incNum], flowEstimatedVal, flowRealVal[flowId]))
                    self.cntrMaster.printAllCntrs (self.logFile, printAlsoVec=True)
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
        self.closeOutputFiles ()

    def closeOutputFiles (self):
        """
        Close the output files
        """
        if VERBOSE_PCL in self.verbose:
            self.pclOutputFile.close ()

        if (VERBOSE_RES in self.verbose):
            self.resFile.close ()
            
        if (VERBOSE_FULL_RES in self.verbose):
            self.fullResFile.close ()

        if self.logFile!=None:
            self.logFile.close ()
                
    def writeProgress (self, infoStr=None):
        """
        If the verbose requires that, report the progress to self.logFile
        """ 
        if not (VERBOSE_PROGRESS in self.verbose):
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
        dict['numIncs']     = self.incNum+1
        dict['mode']        = self.mode
        dict['cntrSize']    = self.cntrSize
        dict['depth']       = self.depth
        dict['width']       = self.width
        dict['numFlows']    = self.numFlows
        dict['seed']        = self.seed
        dict['maxValBy']    = self.maxValBy
        dict['dwnSmpl']     = self.dwnSmpl
        return dict
    
def LaunchCmsSim (
        traceName   : str = 'Rand', 
        mode            : str = 'AEE_ds',
        cntrSize        : int = 4, 
        width           : int = 2,
    ):    
    """
    """   
    depth = 4
    if traceName=='Rand':
        cms = CountMinSketch (
            width           = 2, 
            depth           = 2,
            numCntrsPerBkt  = 2,
            mode            = mode, 
            traceName   = traceName,
            numEpsilonStepsIceBkts  = 5, 
            numEpsilonStepsInRegBkt = 2,
            numEpsilonStepsInXlBkt  = 5,
            verbose                 = [VERBOSE_LOG, VERBOSE_LOG_DWN_SMPL], #, VERBOSE_LOG_END_SIM], #[VERBOSE_LOG_SHORT, VERBOSE_LOG_DWN_SMPL, VERBOSE_LOG_END_SIM], # VERBOSE_LOG_DWN_SMPL, VERBOSE_LOG_END_SIM, VERBOSE_LOG_END_SIM, VERBOSE_LOG, VERBOSE_DETAILS
            numOfExps               = 2, 
            maxNumIncs              = 1000,
            maxValBy                = 'int',
            cntrSize                = 5, 
        )
        cms.sim (
            numFlows = 4,
            )
    else:
        cms = CountMinSketch (
            maxValBy        = None, 
            width           = width,
            depth           = depth,
            mode            = mode,
            numCntrsPerBkt  = 1, #16
            cntrSize        = cntrSize, 
            traceName       = traceName,
            numEpsilonStepsIceBkts  = 6, 
            numEpsilonStepsInRegBkt = 5,
            numEpsilonStepsInXlBkt  = 7,
            numOfExps               = 10,
            maxNumIncs              = 100000000, 
            verbose                 = [VERBOSE_LOG_END_SIM, VERBOSE_LOG_DWN_SMPL, VERBOSE_RES, VERBOSE_PCL], 
        )
        cms.sim ()

def runMultiProcessSim ():
    """
    Generate a multi-process simulation. 
    This func uses fork(), and therefore can run only in UNIX environment.
    """
    mode = 'AEE_ds'     
    for trace in ['Rand']: #['Caida1', 'Caida2']:
        for width in [10, 12]: #[2**i for i in range (10, 19)]: 
            pid = os.fork ()
            if pid: # parent
                print (f'Launched cms process for trace={trace}, width={width}')
                continue
            
            # now we know that this is a child process
            LaunchCmsSim (
                cntrSize = 8,
                mode     = mode,
                width    = width,
            )
  
if __name__ == '__main__':
    try:
        mode = 'F2P_li_h2' 
        for traceName in ['Caida2']: #['Caida2']: #, 'Caida2']: 
            for width in [2**i for i in range (18, 19)]:   
                LaunchCmsSim (
                    traceName   = traceName,
                    cntrSize    = 8,
                    mode        = mode,
                    width       = width,
                )
                
    except KeyboardInterrupt:
        print ('Keyboard interrupt.')
