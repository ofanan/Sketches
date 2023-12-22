import matplotlib #, seaborn
import matplotlib.pyplot as plt
# import matplotlib.ticker
# import matplotlib.pylab as pylab

import math, random, os, pickle, mmh3, time, csv
import numpy as np
from datetime import datetime
import settings, PerfectCounter, Buckets, NiceBuckets

from printf import printf, printarFp

class CountMinSketch:

    # given the flowId and the row, return the hash, namely, the corresponding counter in that row  
    hashOfFlow = lambda self, flowId, row : mmh3.hash(str(flowId), seed=self.seed + row) % self.width

    # given the flowId, return the list of cntrs hashed to this flow Id.   
    hashedCntrsOfFlow = lambda self, flowId : [self.mat2aridx(row, mmh3.hash(str(flowId), seed=self.seed + row) % self.width) for row in range(self.depth)] 

    # given the row and col. in a matrix, return the corresponding index if the mat is flattened into a 1D array.
    mat2aridx  = lambda self, row, col       : self.width*row + col 

    genSettingsStr = lambda self : f'cms_{self.mode}_d{self.depth}_w{self.width}_f{self.numFlows}_bit{self.cntrSize}'
    
    def __init__(self,
                 width          = 2, # the number of counters per row.
                 depth          = 2, # the number of rows or hash functions.
                 numCntrsPerBkt = 2,
                 numFlows       = 10, # the total number of flows to be estimated.
                 mode           = 'PerfectCounter', # the counter mode (e.g., SEC, AEE, realCounter).
                 cntrSize       = 2, # num of bits in each counter
                 verbose        = [],
                 seed           = settings.SEED,
                 ):
        
        """
        """
        random.seed (settings.SEED)
        self.numCntrsPerBkt = int(numCntrsPerBkt)
        if depth<1 or width<1 or cntrSize<1:
            settings.error (f'CountMinSketch was called with depth={depth}, width={width}, cntrSize={cntrSize}. All these parameters should be at least 1.')
        if depth<2 or width<2:
            print (f'Note: CountMinSketch was called with depth={depth} and width={width}.')            
        self.cntrSize, self.width, self.depth, self.numFlows = cntrSize, width, depth, numFlows
        
        self.mode, self.seed,  = mode, seed
        self.numCntrs   = self.width * self.depth
        
        numBucketsFP = self.numCntrs / self.numCntrsPerBkt
        self.numBuckets = int(numBucketsFP)
        if numBucketsFP != self.numBuckets:
            settings.error (f'You chose numCntrs={self.numCntrs}, numCntrsPerBkt={self.numCntrsPerBkt}. However, numCntrs should be divisible by numCntrsPerBkt')           
        if self.numBuckets < 1:
            settings.error (f'numOfBkts={self.numOfBkts}')
        self.conf       = settings.getConfByCntrSize (cntrSize=self.cntrSize)
        self.verbose = verbose
        self.genOutputDirectories ()

    def genCntrMaster (self):
        """
        Generate self.cntrMaster according to the mode requested
        """
        if self.mode=='PerfectCounter':
             self.cntrMaster = PerfectCounter.CntrMaster (
                                    cntrSize    = self.cntrSize, 
                                    numCntrs    = self.numCntrs, 
                                    verbose     = self.verbose)
        elif self.mode=='NiceBuckets':
            self.cntrMaster = NiceBuckets.CntrMaster (
                                    cntrSize                = self.cntrSize, 
                                    numCntrs                = self.numCntrs, 
                                    numCntrsPerRegBkt       = self.numCntrsPerBkt,
                                    numCntrsPerXlBkt        = self.numCntrsPerBkt,
                                    numEpsilonStepsInRegBkt = 4,
                                    numEpsilonStepsInXlBkt  = 8, 
                                    numXlBkts               = self.width,
                                    verbose                 = self.verbose)
        elif self.mode=='SecBuckets':
             self.cntrMaster = Buckets.Buckets (
                                    cntrSize        = self.cntrSize, 
                                    numCntrs        = self.numCntrs, 
                                    numCntrsPerBkt  = self.numCntrsPerBkt, 
                                    mode            = 'SEC', 
                                    verbose         = self.verbose)
        elif self.mode=='IceBuckets':
            self.cntrMaster = Buckets.Buckets (
                                    cntrSize        = self.cntrSize, 
                                    numCntrs        = self.numCntrs, 
                                    numCntrsPerBkt  = self.numCntrsPerBkt, 
                                    mode            = 'ICE',
                                    numEpsilonSteps = 5,
                                    cntrMaxVal      = (1 << self.cntrSize) - 1,
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
            print(f'Sorry, the mode {self.mode} that you requested is not supported')

    
    def genOutputDirectories (self):
        """
        Generate and open the directories for output (results, logs, etc.):
        - Generate directories for the output files if not exist
        - Verify the verbose level requested.
        """      
        if settings.VERBOSE_DETAILED_RES in self.verbose or settings.VERBOSE_FULL_RES in self.verbose:
            self.verbose.append (settings.VERBOSE_RES)
        # if not (settings.VERBOSE_PCL in self.verbose):
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
        if (settings.VERBOSE_PCL in self.verbose):
            pickle.dump(dict, self.pclOutputFile) 
    
    def writeDictToResFile (self, dict):
        """
        Write a single dict of data into resOutputFile
        """
        if (settings.VERBOSE_RES in self.verbose):
            printf (self.resFile, f'{dict}\n\n') 
        if (settings.VERBOSE_FULL_RES in self.verbose):
            printf (self.fullResFile, f'{dict}\n\n') 
    
    def calcRmseStat (self) -> dict: 
        """
        Calculate and potentially print to .log and/or .res file (based on self.verbose) the RMSE statistics based on the values measured and stored in self.cntrMaster.sumSqEr.
        Return a dict of the calculated data.  
        """
        
        self.numOfExps = self.expNum + 1 # Allow writing intermmediate results. Assume we began with expNum=0.
        Rmse     = [math.sqrt (self.sumSqEr[expNum]/self.numIncs) for expNum in range(self.numOfExps)]
        normRmse = [Rmse[expNum]/self.numIncs  for expNum in range(self.numOfExps)]
        if (settings.VERBOSE_LOG in self.verbose):
            printf (self.logFile, '\nnormRmse=')
            printarFp (self.logFile, normRmse)
        
        normRmseAvg          = np.average    (normRmse)
        normRmseConfInterval = settings.confInterval (ar=normRmse, avg=normRmseAvg)
        return {'numOfExps'     : self.numOfExps,
                'numIncs'     : self.numIncs,
                'mode'          : self.mode,
                'cntrSize'      : self.cntrSize, 
                'depth'         : self.depth,
                'width'         : self.width,
                'numFlows'      : self.numFlows,
                'seed'          : self.seed,
                'Avg'           : normRmseAvg,
                'normRmse'      : normRmse,
                'Lo'            : normRmseConfInterval[0],
                'Hi'            : normRmseConfInterval[1]}

    def openOutputFiles (self) -> None:
        """
        Open the output files (.res, .log, .pcl), as defined by the verbose level requested.
        """      
        if settings.VERBOSE_PCL in self.verbose:
            self.pclOutputFile = open(f'../res/pcl_files/sim_{settings.getMachineStr()}.pcl', 'ab+')

        if (settings.VERBOSE_RES in self.verbose):
            self.resFile = open (f'../res/cms_{settings.getMachineStr()}.res', 'a+')
            
        if (settings.VERBOSE_FULL_RES in self.verbose):
            self.fullResFile = open (f'../res/cms_full.res', 'a+')
            
    def sim (self, 
             numIncs        = 5000, # overall number of increments (# of pkts in the trace) 
             numOfExps      = 1,  # number of repeated experiments. Relevant only for randomly-generated traces.
             traceFileName  = None
             ):
        """
        Simulate the count min sketch
        """
        
        self.openOutputFiles ()
        self.numIncs, self.numOfExps = numIncs, numOfExps
        
        if traceFileName==None:
            flowRealVal     = [0] * self.numFlows
            self.sumSqEr    = [0] * self.numOfExps # self.sumSqEr[j] will hold the sum of the square errors collected at experiment j. 

            print ('Started running random input sim at t={}. mode={}, cntrSize={}, depth={}, width={}, numFlows={}' .format (
                    datetime.now().strftime('%H:%M:%S'), self.mode, self.cntrSize, self.depth, self.width, self.numFlows))

            for self.expNum in range (self.numOfExps):
                self.writeProgress () # log the beginning of the experiment; used to track the progress of long runs.
                self.genCntrMaster ()
    
                if (settings.VERBOSE_LOG in self.verbose or settings.VERBOSE_PROGRESS in self.verbose):
                    infoStr = '{}_{}' .format (self.genSettingsStr(), self.cntrMaster.genSettingsStr())
                    self.logFile = open (f'../res/log_files/{infoStr}.log', 'a+')
                    self.cntrMaster.setLogFile(self.logFile)
                
                for incNum in range(self.numIncs):
                    flowId = math.floor(np.random.exponential(scale = 2*math.sqrt(self.numFlows))) % self.numFlows
                    # flowId = mmh3.hash(str(flowId)) % self.numFlows
                    flowRealVal[flowId]     += 1
                    flowEstimatedVal   = self.incNQueryFlow (flowId=flowId)
                    self.sumSqEr[self.expNum] += (((flowRealVal[flowId] - flowEstimatedVal)/flowRealVal[flowId])**2)                
                    if settings.VERBOSE_LOG in self.verbose:
                        self.cntrMaster.printAllCntrs (self.logFile)
                        printf (self.logFile, ' hashes={}, estimatedVal={:.0f} realVal={:.0f} \n' .format(self.hashedCntrsOfFlow(flowId), flowEstimatedVal, flowRealVal[flowId])) 
                if settings.VERBOSE_FULL_RES in self.verbose:
                    printf (self.fullResFile, f'{self.calcRmseStat()}\n\n') 
            dict = self.calcRmseStat    ()
            if settings.VERBOSE_PCL in self.verbose:
                self.dumpDictToPcl    (dict)
            if settings.VERBOSE_RES in self.verbose:
                printf (self.resFile, f'{dict}\n\n') 

        else: # read trace from a file
            incNum          = 0
            flowRealVal     = [0] * self.numFlows
            self.sumSqEr    = 0 # self.sumSqEr will hold the sum of the square errors.  
            relativePathToInputFile = f'{settings.getTracesPath()}Caida/{traceFileName}'
            settings.checkIfInputFileExists (relativePathToInputFile)
            csvFile = open (relativePathToInputFile, 'r')
            csvReader = csv.reader(csvFile) #, delimiter=' ', quotechar='|')
            self.genCntrMaster ()
            if settings.VERBOSE_LOG in self.verbose:
                infoStr = '{}_{}' .format (self.genSettingsStr(), self.cntrMaster.genSettingsStr())
                self.logFile = open (f'../res/log_files/{infoStr}.log', 'w')
                self.cntrMaster.setLogFile(self.logFile)

            print ('Started running trace input sim at t={}. mode={}, cntrSize={}, depth={}, width={}, numFlows={}' .format (
                    datetime.now().strftime('%H:%M:%S'), self.mode, self.cntrSize, self.depth, self.width, self.numFlows))
            for row in csvReader:
                flowId = int(row[0]) % self.numFlows
                if incNum==self.numIncs:
                    break
                flowRealVal[flowId]     += 1
                
                flowEstimatedVal   = self.incNQueryFlow (flowId=flowId)
                self.sumSqEr += (((flowRealVal[flowId] - flowEstimatedVal)/flowRealVal[flowId])**2)
                incNum  += 1                
                if settings.VERBOSE_LOG in self.verbose:
                    self.cntrMaster.printAllCntrs (self.logFile)
                    printf (self.logFile, ' hashes={}, estimatedVal={:.0f} realVal={:.0f} \n' .format(self.hashedCntrsOfFlow(flowId), flowEstimatedVal, flowRealVal[flowId])) 

            Rmse     = math.sqrt (self.sumSqEr/self.numIncs)
            normRmse = Rmse/self.numIncs
            if (settings.VERBOSE_LOG in self.verbose):
                printf (self.logFile, f'\nnormRmse={normRmse}')
            
            dict = {'numOfExps'     : 1,
                    'numIncs'       : self.numIncs,
                    'mode'          : self.mode,
                    'cntrSize'      : self.cntrSize, 
                    'depth'         : self.depth,
                    'width'         : self.width,
                    'numFlows'      : self.numFlows,
                    'seed'          : self.seed,
                    'Avg'           : normRmse}
            if settings.VERBOSE_PCL in self.verbose:
                self.dumpDictToPcl    (dict)
            if settings.VERBOSE_RES in self.verbose:
                printf (self.resFile, f'{dict}\n\n') 
                
    def collectStatOfTrace (self, 
             numIncs        = float('inf'), # overall number of increments (# of pkts in the trace) 
             traceFileName  = None
             ):
        """
        Collect statistics about a trace
        """
        
        self.numIncs    = numIncs
        
        if traceFileName==None:
            flowRealVal     = [0] * self.numFlows

            for incNum in range(self.numIncs):
                flowId = math.floor(np.random.exponential(scale = 2*math.sqrt(self.numFlows))) % self.numFlows
                # flowId = mmh3.hash(str(flowId)) % self.numFlows
                flowRealVal[flowId]     += 1
        else: # read trace from a file
            incNum          = 0
            flowRealVal     = [0] * self.numFlows
            relativePathToInputFile = f'{settings.getTracesPath()}Caida/{traceFileName}'
            settings.checkIfInputFileExists (relativePathToInputFile)
            csvFile = open (relativePathToInputFile, 'r')
            csvReader = csv.reader(csvFile) #, delimiter=' ', quotechar='|')
            for row in csvReader:
                flowId = int(row[0]) % self.numFlows
                flowRealVal[flowId]     += 1
                incNum          += 1
                if incNum==self.numIncs:
                    break
        
        outputFileName = 'rand' if traceFileName==None else traceFileName + f'_{incNum}incs'
        outputFile = open (f'../res/{outputFileName}.txt', 'w')
        maxFlowSize = max (flowRealVal)
        numBins = min (100, maxFlowSize+1)
        binSize = maxFlowSize // (numBins-1)
        binVal  = [None] * numBins 
        for bin in range(numBins):
            binVal[bin] = len ([flowId for flowId in range(self.numFlows) if (flowRealVal[flowId]//binSize)==bin])
        binFlowSizes = [binSize*bin for bin in range (numBins)]
        print (f'binYVals={binVal}')
        printf (outputFile, f'numFlows={self.numFlows}, num zero flows={len ([item for item in flowRealVal if item==0])}, num non-zeros flows={len ([item for item in flowRealVal if item>0])}')
        printf (outputFile, f'\nmaxFlowSize={maxFlowSize}, binVal={binVal}')
        printf (outputFile, f'\nbinFlowSizes={binFlowSizes}')
        printf (outputFile, f'\nflowSizes={flowRealVal}')
        _, ax = plt.subplots()
        ax.plot ([binSize*bin for bin in range (numBins)], binVal)
        ax.set_yscale ('log')
        plt.show ()
        plt.savefig (f'../res/{outputFileName}.pdf', bbox_inches='tight')        
        
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
    
def main(mode, runShortSim=True):
    """
    """   
    traceFileName           = 'equinix-nyc.dirB.20181220-140100.UTC.anon.pcap.csv'
    
    if runShortSim:
        width, depth, cntrSize  = 2, 2, 4
        numFlows                = width*depth*1
        numCntrsPerBkt          = 2
        numIncs                 = 100000 #(width * depth * cntrSize**3)/2
        numOfExps               = 1
        verbose                 = [] #settings.VERBOSE_LOG, settings.VERBOSE_DETAILS
    else:
        width, depth, cntrSize  = 64, 4, 8
        numFlows                = width*depth*16
        numCntrsPerBkt          = 16
        numIncs                 = 100000000 #(width * depth * cntrSize**3)/2
        numOfExps               = 1
        verbose                 = [settings.VERBOSE_RES] # settings.VERBOSE_RES, settings.VERBOSE_FULL_RES, settings.VERBOSE_PCL] # settings.VERBOSE_LOG, settings.VERBOSE_RES, settings.VERBOSE_PCL, settings.VERBOSE_DETAILS
         
    cms = CountMinSketch (width=width, depth=depth, cntrSize=cntrSize, numFlows=numFlows, verbose=verbose, 
                          numCntrsPerBkt = numCntrsPerBkt, 
                          mode=mode)
    cms.sim (numOfExps=numOfExps, numIncs=numIncs, traceFileName=traceFileName)
    # cms.collectStatOfTrace(traceFileName=traceFileName) #, numIncs=100)
    
if __name__ == '__main__':
    main (mode='NiceBuckets', runShortSim=False)
