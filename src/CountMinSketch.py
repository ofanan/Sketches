import matplotlib 
import matplotlib.pyplot as plt
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
        elif self.mode=='IceBuckets':
            self.cntrMaster = Buckets.Buckets (
                                    cntrSize        = self.cntrSize, 
                                    numCntrs        = self.numCntrs, 
                                    numCntrsPerBkt  = self.numCntrsPerBkt, 
                                    mode            = 'ICE',
                                    numEpsilonSteps = 6,
                                    verbose         = self.verbose)
        elif self.mode=='NiceBuckets':
            self.cntrMaster = NiceBuckets.CntrMaster (
                                    cntrSize                = self.cntrSize, 
                                    numCntrs                = self.numCntrs, 
                                    numCntrsPerRegBkt       = self.numCntrsPerBkt,
                                    numCntrsPerXlBkt        = self.numCntrsPerBkt,
                                    numEpsilonStepsInRegBkt = 4,
                                    numEpsilonStepsInXlBkt  = 8, 
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
    
                if (settings.VERBOSE_LOG in self.verbose) or (settings.VERBOSE_PROGRESS in self.verbose) or (settings.VERBOSE_LOG_END_SIM in self.verbose):
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
            if (settings.VERBOSE_LOG in self.verbose) or (settings.VERBOSE_LOG_END_SIM in self.verbose):
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

            if settings.VERBOSE_LOG_END_SIM in self.verbose:
                self.cntrMaster.printCntrsStat (self.logFile) 
                # self.cntrMaster.printAllCntrs (self.logFile)
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
        numIncs                 = 4945 #(width * depth * cntrSize**3)/2
        numOfExps               = 1
        verbose                 = [settings.VERBOSE_LOG_END_SIM] #settings.VERBOSE_LOG_END_SIM, settings.VERBOSE_LOG, settings.VERBOSE_DETAILS
    else:
        width, depth, cntrSize  = 1024, 4, 8
        numFlows                = 4096 # width*depth*16
        numCntrsPerBkt          = 16
        numIncs                 = float ('inf')   
        numOfExps               = 1
        verbose                 = [settings.VERBOSE_LOG_END_SIM] # settings.VERBOSE_RES, settings.VERBOSE_FULL_RES, settings.VERBOSE_PCL] # settings.VERBOSE_LOG, settings.VERBOSE_RES, settings.VERBOSE_PCL, settings.VERBOSE_DETAILS
         
    cms = CountMinSketch (width=width, depth=depth, cntrSize=cntrSize, numFlows=numFlows, verbose=verbose, 
                          numCntrsPerBkt = numCntrsPerBkt, 
                          mode=mode)
    cms.sim (numOfExps=numOfExps, numIncs=numIncs, traceFileName=traceFileName)
    # cms.collectStatOfTrace(traceFileName=traceFileName) #, numIncs=100)
    
# equinixFlowSizes = [4372, 1307, 3090, 5230, 2069, 5536, 3191, 2740, 4287, 3492, 1982, 1651, 4502, 1488, 1988, 2476, 2204, 2524, 3561, 4143, 3197, 3266, 8462, 2431, 2141, 3950, 2342, 4891, 3904, 4548, 2253, 2868, 2062, 2463, 1805, 1954, 1653, 7677, 4234, 3438, 2838, 2952, 2359, 1719, 1836, 2661, 1941, 2137, 2915, 2331, 1590, 1759, 3764, 2023, 1468, 1957, 3511, 2871, 1770, 2147, 7237, 2421, 2829, 2128, 2751, 2451, 3871, 2163, 4212, 2240, 4425, 4924, 8507, 3877, 2520, 1998, 1690, 2635, 3268, 2536, 2718, 3002, 3165, 1607, 6837, 1947, 2279, 2026, 2392, 2778, 3662, 3161, 2444, 2403, 2897, 6718, 1113, 2715, 2582, 2784, 4778, 4560, 2445, 2032, 2515, 2808, 3107, 3270, 2352, 4652, 2290, 1711, 2689, 8058, 3038, 2337, 2587, 2024, 1976, 2142, 3197, 2203, 3463, 1949, 3608, 2654, 2164, 2568, 2319, 2273, 8001, 4310, 1758, 3740, 1948, 2737, 1826, 3366, 5041, 12002, 2411, 2616, 1921, 2674, 2461, 3834, 2199, 4205, 3323, 4882, 3070, 2593, 1299, 2998, 8573, 2944, 2825, 2387, 2619, 3430, 2143, 8221, 2427, 1781, 4241, 2825, 1964, 2518, 3106, 3701, 3397, 1970, 3020, 2834, 2440, 2327, 2564, 1889, 3570, 3836, 2436, 4207, 3160, 1289, 2830, 3271, 1844, 6849, 1681, 1954, 2949, 1421, 3932, 1705, 1447, 4647, 2406, 1693, 2301, 3321, 8455, 1299, 1835, 1831, 2639, 1602, 2277, 4209, 2203, 6340, 3244, 2189, 2766, 4186, 1991, 6390, 3063, 3711, 2047, 2214, 2129, 5538, 3435, 6050, 3799, 1924, 5063, 3887, 2956, 2664, 2649, 2844, 2782, 2613, 5490, 2636, 1204, 5223, 2338, 2523, 3395, 2356, 1400, 4227, 2710, 4654, 3850, 1540, 1874, 4424, 2540, 2350, 9578, 3124, 3189, 6972, 2111, 6056, 1929, 5765, 2145, 1625, 6189, 6642, 2406, 1902, 2975, 3380, 4625, 2480, 4055, 4380, 2758, 1955, 2049, 2491, 2969, 12763, 2185, 5094, 4718, 3678, 3368, 3134, 1962, 5023, 4111, 2285, 4154, 2385, 3050, 5026, 3407, 3129, 3954, 2603, 2862, 2676, 5457, 2096, 3597, 2297, 4238, 3461, 2647, 3325, 1862, 1957, 2666, 7249, 2289, 5019, 3313, 2333, 3069, 5082, 2698, 1960, 3420, 5258, 2672, 2231, 2456, 1384, 2066, 1270, 4941, 1998, 3371, 2743, 1390, 2476, 3046, 4332, 1909, 3566, 3096, 1409, 2062, 2642, 2078, 1607, 1457, 4257, 2792, 3202, 2709, 1911, 4727, 4108, 3154, 3079, 1876, 1695, 3003, 2637, 2305, 2927, 2909, 3906, 1662, 4656, 3244, 2187, 1723, 5037, 2558, 2291, 3085, 5462, 3230, 1980, 2382, 3011, 2974, 3113, 3653, 5470, 1930, 2966, 3663, 2303, 2630, 1986, 3632, 2707, 4121, 2925, 2069, 5847, 2823, 2856, 3402, 3972, 3581, 1934, 2380, 1334, 2664, 2846, 2445, 4915, 3311, 5608, 3872, 1413, 1237, 2622, 2176, 5471, 2406, 3560, 5196, 3731, 2933, 4111, 1833, 2686, 3800, 3261, 4188, 2827, 2903, 3998, 1519, 1801, 1877, 3008, 3354, 5181, 2370, 4693, 3143, 2489, 4066, 1526, 2278, 2109, 2728, 5057, 2425, 2138, 8791, 2313, 5298, 1853, 4261, 2014, 2230, 5970, 2486, 3015, 1599, 3333, 2718, 3789, 3798, 3105, 2184, 3855, 3035, 2695, 3064, 6078, 1721, 2675, 2879, 2299, 2937, 5199, 2190, 2784, 3439, 2532, 3490, 1984, 2520, 2735, 3388, 3274, 3621, 4338, 5741, 4577, 2965, 3348, 1903, 4347, 2343, 3717, 3147, 4608, 1638, 2632, 4292, 1694, 5611, 3629, 6407, 1772, 2211, 2521, 1835, 2004, 1739, 3580, 3923, 1979, 2357, 2031, 1994, 2724, 1775, 3332, 5503, 3002, 1737, 3525, 1912, 2122, 2194, 2751, 1341, 3605, 2701, 2099, 2076, 2593, 4245, 5505, 1423, 2621, 2072, 2198, 7591, 3616, 3150, 13436, 2815, 3549, 2054, 2164, 2139, 91720, 2157, 5219, 2342, 1105, 2049, 2431, 2892, 3323, 4875, 3051, 2950, 3437, 1727, 4206, 2984, 3879, 3025, 1796, 2124, 2939, 2089, 2857, 3056, 4011, 2929, 5749, 1781, 3350, 4706, 3454, 5307, 2395, 6151, 2365, 2183, 2701, 2632, 3005, 9412, 3041, 6730, 2401, 2243, 2197, 1556, 2636, 2591, 3073, 5676, 5144, 5055, 5727, 4466, 3165, 1610, 2864, 4295, 2436, 4307, 5899, 1449, 2273, 4592, 3202, 5054, 3678, 3973, 2594, 5408, 8612, 4468, 2953, 2154, 2388, 3647, 4582, 2317, 6344, 3726, 3388, 3225, 2490, 2276, 5580, 3966, 1868, 1977, 5436, 3712, 3481, 2852, 3783, 2364, 3420, 2251, 1902, 3056, 2580, 1888, 2363, 1620, 2144, 3557, 2206, 1570, 1818, 3411, 1526, 4336, 4796, 2047, 2384, 4548, 2150, 1690, 1533, 3261, 1939, 3977, 3867, 4043, 3943, 1878, 3496, 3005, 3664, 2251, 2559, 3040, 1898, 2188, 2022, 7628, 4182, 2556, 2167, 2504, 2446, 4106, 1750, 3620, 3001, 1824, 6639, 2587, 2530, 4219, 2899, 2401, 2798, 3061, 1518, 1762, 2138, 2857, 2853, 1905, 2930, 3552, 2475, 2193, 2511, 3537, 4074, 1758, 3540, 3683, 2542, 2429, 3959, 1816, 2678, 2924, 2231, 2406, 2321, 2104, 2341, 6114, 3192, 2270, 4246, 1040, 3979, 4101, 5654, 1902, 3430, 5865, 1816, 3183, 3052, 2187, 5450, 5537, 3905, 2095, 2893, 3376, 2381, 2552, 2312, 1462, 2892, 1585, 2089, 1139, 1593, 2095, 2150, 3198, 3136, 3790, 4763, 2164, 1798, 2892, 3895, 2685, 6828, 2300, 3896, 1920, 4259, 2851, 6200, 3864, 2083, 6555, 3583, 2912, 3770, 2472, 1439, 1972, 1920, 2483, 3476, 2231, 2508, 2639, 2970, 8346, 9949, 1858, 2242, 1946, 3031, 2848, 2781, 3245, 1043, 2760, 12017, 2179, 2498, 2134, 2797, 2747, 3241, 2678, 3575, 1707, 3694, 2103, 3388, 2827, 3302, 1961, 2632, 2987, 1941, 4004, 3265, 4964, 2248, 4267, 1852, 4822, 4421, 2581, 3065, 9949, 2106, 2959, 4046, 4260, 2785, 2222, 12723, 1664, 3443, 3969, 2517, 3390, 7984, 7833, 3234, 3840, 2766, 1942, 4202, 1430, 4255, 2580, 5082, 2859, 3172, 5856, 2745, 2986, 2194, 3810, 2669, 4845, 2328, 3769, 2243, 5222, 2506, 2030, 4116, 2500, 2224, 3204, 2081, 3073, 2863, 5919, 6440, 5209, 3988, 1733, 3883, 4928, 2899, 2936, 3909, 1941, 1426, 3597, 2583, 13998, 2701, 1474, 4674, 3786, 3455, 1872, 1754, 1558, 3738, 4675, 2841, 6206, 1823, 4277, 2267, 4465, 2047, 2586, 4053, 1165, 3613, 4877, 2418, 2929, 2349, 1782, 1906, 1844, 2809, 2605, 2929, 3517, 4919, 3457, 2304, 3411, 2217, 6518, 1730, 3347, 3111, 2969, 2817, 4748, 2724, 2007, 1773, 4917, 3723, 4875, 2705, 3042, 2277, 3508, 2820, 1888, 1908, 2164, 2171, 4552, 1617, 2756, 2459, 1840, 2451, 2129, 3111, 3786, 2337, 3723, 1633, 4827, 3376, 2228, 2316, 1825, 3742, 3366, 1823, 2884, 4217, 1772, 2359, 2717, 2981, 4569, 1849, 2863, 2069, 2161, 2569, 3106, 2310, 1564, 2535, 2033, 1523, 3120, 4649, 2094, 2424, 3000, 2557, 2258, 4234, 2010, 4382, 1279, 3820, 1462, 1689, 3367, 4612, 2004, 1615, 3993, 1703, 3196, 3191, 4018, 4897, 1542, 3388, 3030, 2572, 2702, 3248, 24960, 2149, 2743, 7750, 1965, 2498, 9175, 2656, 3751, 2083, 2570, 3530, 3168, 3166, 1272, 2328, 2893, 3117, 3794, 4568, 7275, 3196, 2526, 3370, 3042, 4446, 3278, 1915, 3677, 2531, 2830, 2929, 1628, 2580, 2486, 2817, 1996, 2525, 3048, 3088, 7375, 2685, 2139, 1887, 2319, 5825, 3246, 3118, 2811, 2652, 2510, 2161, 2541, 3424, 2159, 5748, 4037, 3669, 1798, 2631, 2484, 4340, 3464, 2656, 3221, 2642, 1666, 2245, 4288, 1815, 3316, 3853, 1830, 3009, 2304, 1679, 4671, 2926, 1754, 1792, 1595, 2918, 1456, 3305, 2808, 4114, 4163, 4418, 3581, 4253, 2886, 1487, 3589, 3236, 2187, 3435, 2588, 3189, 3031, 6080, 3063, 2913, 3126, 2961, 4039, 3649, 11571, 4596, 1474, 2063, 2296, 2257, 2518, 2153, 3632, 3781, 2256, 1998, 2366, 2742, 2087, 2802, 3222, 1945, 1146, 2654, 2498, 2853, 2644, 2658, 2538, 2383, 2990, 2687, 5629, 3648, 2224, 4548, 2707, 2207, 4663, 3248, 2676, 2317, 4575, 4011, 2977, 4568, 2123, 2908, 1923, 7271, 3567, 3592, 2488, 2913, 3216, 4383, 2454, 2543, 1697, 3249, 2508, 6785, 5652, 5367, 3779, 2659, 3128, 2853, 2239, 2196, 2571, 2901, 2149, 1889, 2191, 3515, 2056, 2586, 3823, 1967, 2374, 1745, 3886, 1733, 7175, 3423, 2270, 2819, 3385, 3068, 1290, 2403, 3074, 4567, 3717, 1324, 2509, 2380, 4993, 6662, 3018, 2561, 2511, 3106, 2463, 3844, 2206, 2685, 1998, 2303, 2304, 3483, 4592, 2287, 2438, 3171, 3051, 1941, 1879, 2798, 2973, 2482, 2938, 4865, 3184, 1505, 2784, 2900, 1898, 2292, 2369, 2219, 1926, 1424, 3583, 3697, 5221, 2354, 3338, 1751, 2363, 2941, 6189, 3158, 1689, 2908, 3831, 4223, 1789, 4077, 1604, 3023, 2147, 3099, 2926, 3199, 3092, 2689, 1839, 7029, 1426, 3017, 2053, 2683, 3440, 4607, 4285, 1331, 2145, 2726, 2061, 4293, 6053, 2178, 3155, 2869, 2065, 3737, 2917, 3357, 2169, 1912, 6176, 2370, 5038, 1535, 1968, 2805, 2480, 1731, 3173, 3150, 2266, 3177, 1359, 3147, 3172, 2210, 1421, 1975, 1762, 2178, 3159, 3147, 3861, 2583, 2983, 5436, 3524, 3183, 2454, 2611, 4938, 2798, 4803, 3427, 1622, 4959, 2528, 1469, 2893, 5269, 3233, 3165, 7178, 3435, 2690, 4572, 4988, 2494, 3443, 4415, 4790, 3404, 2750, 2923, 3684, 3550, 2474, 2401, 2356, 2572, 3188, 3969, 6695, 2625, 3136, 1319, 2555, 3410, 5213, 3735, 8731, 2836, 1771, 2483, 2763, 2519, 2289, 3280, 2210, 3388, 2618, 2546, 1507, 3878, 2427, 3559, 3271, 3834, 1574, 3331, 2622, 3021, 2104, 2991, 1977, 2267, 2809, 2732, 4552, 2342, 2280, 1673, 2957, 2928, 3661, 2023, 1766, 2034, 7828, 1970, 2897, 2937, 2202, 2514, 1734, 2234, 10863, 1974, 5956, 2581, 2430, 2691, 3591, 4082, 2162, 2826, 3703, 2375, 3004, 2879, 3178, 2048, 2491, 1937, 3831, 1868, 1585, 1724, 1505, 4553, 2596, 3533, 1216, 2533, 1829, 3437, 1212, 5753, 5228, 1769, 3478, 4023, 2075, 2742, 2893, 3169, 1867, 1676, 4112, 1918, 5080, 2414, 5600, 4810, 4777, 3998, 2688, 1161, 3019, 3762, 6043, 2663, 3148, 6629, 3337, 3525, 2210, 3678, 1955, 2775, 1859, 6430, 3195, 1580, 2574, 1736, 1511, 3427, 3420, 3465, 1985, 4296, 3639, 2749, 2586, 2002, 3162, 3097, 1398, 2392, 1853, 2817, 6045, 7292, 2752, 5859, 1770, 6273, 1750, 4613, 2477, 2457, 2404, 3878, 1948, 2309, 1609, 3888, 1616, 2078, 3363, 3566, 3744, 2757, 2145, 3601, 3439, 3018, 4322, 3630, 2102, 1430, 5588, 5514, 4369, 3371, 6757, 2081, 3525, 2039, 3677, 1835, 3892, 2584, 1910, 3173, 5633, 4167, 3549, 4197, 2506, 5147, 4334, 2534, 2468, 2298, 5764, 1793, 3070, 5318, 1662, 2197, 1417, 4909, 2481, 2296, 4064, 3421, 3366, 2661, 1766, 2571, 2293, 2567, 2078, 1353, 3437, 2621, 2925, 5938, 2769, 2871, 2814, 2886, 1907, 1967, 1841, 2435, 4675, 3215, 2511, 1249, 4879, 7714, 1554, 2662, 4650, 2990, 3130, 6375, 2059, 2465, 2564, 7595, 3604, 1306, 6052, 2934, 2627, 1234, 4467, 1831, 2617, 4229, 5378, 1577, 2562, 4532, 1778, 2125, 5501, 1692, 2315, 5083, 4395, 2722, 3099, 2774, 3324, 2733, 2736, 5247, 1098, 1813, 1745, 2088, 2410, 3834, 2160, 7676, 2629, 1781, 2281, 3910, 3519, 1513, 1929, 1920, 4311, 3679, 3255, 4505, 1421, 4982, 2188, 1969, 2431, 45817, 3088, 2138, 1969, 2466, 1951, 2287, 5607, 2155, 3216, 3188, 3275, 1116, 2515, 3953, 1723, 2644, 2251, 3351, 1916, 3057, 1574, 2579, 1340, 2265, 3419, 2099, 3221, 7656, 3594, 1615, 2509, 5087, 3594, 4689, 2246, 4176, 3104, 4511, 2300, 2591, 1960, 1649, 2678, 4105, 4570, 2846, 3123, 2039, 2458, 3333, 2652, 2992, 2296, 5193, 1891, 1812, 3875, 4077, 6840, 3554, 2628, 2692, 3250, 4909, 4300, 2655, 1714, 3171, 2140, 6541, 1834, 3759, 1564, 4840, 2131, 3777, 2640, 2450, 2955, 5491, 4555, 2808, 1555, 1300, 3524, 2135, 3766, 3760, 2848, 2457, 4009, 4639, 2546, 2126, 3824, 2311, 2621, 2528, 1429, 3608, 1849, 4163, 2655, 5482, 7650, 2179, 2016, 1413, 8659, 5100, 2960, 2583, 5625, 3069, 3186, 2337, 3202, 1913, 5637, 2047, 2332, 13983, 2662, 3323, 6137, 2896, 1861, 7322, 2349, 2113, 1720, 2623, 1706, 1919, 4285, 1971, 2414, 2640, 4308, 4073, 1954, 1814, 2266, 2125, 1977, 1676, 2226, 4050, 1779, 4088, 4665, 3302, 1541, 2772, 1876, 3725, 3369, 2599, 2052, 1828, 2515, 2281, 4189, 2536, 2801, 1773, 5059, 3723, 1517, 1356, 2996, 2440, 2382, 1379, 2185, 2961, 2597, 3286, 2121, 3056, 3432, 1943, 3991, 5800, 2547, 1042, 2603, 5002, 2417, 1989, 2045, 2047, 2899, 2588, 2617, 2051, 3008, 2310, 4001, 2575, 2795, 5877, 2718, 2328, 2362, 3036, 2681, 1523, 2449, 2125, 2554, 11377, 7588, 2986, 3112, 6361, 2407, 3548, 2064, 5850, 2159, 2603, 1426, 2321, 2002, 3097, 2540, 1629, 2834, 2166, 4065, 1959, 6563, 1800, 2440, 3284, 2604, 4471, 4807, 2735, 2730, 2536, 1797, 3154, 3882, 5531, 2226, 3873, 2722, 2364, 3417, 3284, 2908, 3141, 2167, 3007, 3963, 2451, 1618, 1561, 2298, 2641, 2929, 3843, 1906, 2128, 3160, 2880, 5378, 2515, 4852, 2385, 2530, 2136, 3823, 5699, 2503, 1651, 2743, 2612, 3891, 1834, 3646, 1970, 2575, 1770, 1697, 2787, 2071, 2921, 5605, 6033, 2392, 2121, 5540, 1712, 2053, 5510, 3095, 1524, 1623, 5079, 2136, 1947, 1512, 2979, 4439, 1431, 2476, 2202, 1324, 3825, 2049, 3511, 1674, 4130, 2178, 1883, 5263, 1749, 4398, 3228, 3830, 2904, 4145, 2850, 1944, 1255, 1387, 1938, 5580, 1392, 5243, 2988, 3565, 3015, 2213, 1610, 2678, 5540, 4400, 3139, 1934, 1931, 1719, 2674, 2787, 3961, 3152, 1800, 3165, 2044, 3671, 2133, 3495, 2138, 4226, 3367, 2352, 1340, 2105, 3184, 1526, 3855, 3853, 2303, 1575, 3910, 2188, 6029, 2553, 4199, 1987, 3442, 3085, 2308, 1929, 2465, 1687, 2507, 1829, 3348, 2959, 1665, 2128, 3813, 13331, 4325, 2478, 1894, 2958, 2942, 5989, 2689, 1278, 2196, 2027, 2302, 2236, 2176, 1470, 2576, 2727, 1801, 2965, 3674, 2329, 3306, 1664, 8294, 1883, 2964, 3899, 2959, 3563, 1909, 3140, 4436, 4512, 4921, 2561, 4206, 5456, 3992, 2064, 3556, 2711, 3251, 3623, 1809, 4179, 5417, 4944, 6530, 7778, 1342, 3333, 2139, 4895, 1609, 5222, 4741, 1781, 5038, 7982, 4616, 2746, 4567, 1394, 8169, 3578, 3165, 2015, 2240, 2544, 2220, 2966, 5857, 2640, 4342, 2514, 2213, 2203, 2354, 2216, 4968, 1968, 2479, 10886, 3107, 2958, 2405, 2269, 2752, 2718, 3764, 3615, 2094, 3985, 2732, 3385, 2644, 2355, 2557, 2645, 2444, 2976, 2649, 1607, 2274, 7830, 2470, 1575, 3009, 2619, 1840, 6416, 3115, 2987, 4334, 2128, 2642, 3589, 5250, 1857, 4206, 3707, 6256, 4062, 1914, 1659, 5896, 4551, 2565, 3304, 2727, 2214, 4719, 3952, 2258, 2386, 3281, 3452, 2254, 3900, 3164, 2622, 3463, 2678, 4534, 3130, 4632, 2102, 1685, 3934, 3805, 1851, 2694, 3369, 3108, 2727, 2381, 1701, 4636, 3293, 4193, 2658, 5299, 2389, 2610, 3448, 2636, 3562, 2696, 2752, 2163, 2767, 3296, 2132, 2877, 1976, 2566, 3254, 2854, 1301, 2815, 1983, 6035, 3364, 4006, 1308, 1636, 974, 3162, 5779, 4550, 1819, 5021, 1732, 5117, 2543, 5857, 2582, 1635, 2723, 3336, 5890, 2395, 2554, 3663, 1497, 5089, 2957, 5084, 2312, 2570, 1610, 2330, 4076, 6065, 2959, 2491, 3061, 5328, 2531, 1716, 3432, 2828, 4014, 2134, 2802, 3730, 2206, 2398, 4326, 1735, 3238, 2191, 5315, 2052, 3539, 3839, 7730, 2280, 3866, 1978, 3028, 3985, 2185, 2297, 1354, 5387, 2601, 3224, 2261, 2474, 2167, 1959, 4682, 1940, 3110, 7183, 2784, 3472, 2347, 3851, 3406, 3158, 1944, 4052, 2743, 3376, 2893, 1547, 7977, 2268, 2004, 4113, 1545, 2145, 2424, 6036, 2000, 2754, 3356, 4437, 3120, 5875, 2448, 5246, 5709, 3653, 3540, 3050, 2465, 3244, 2337, 4924, 3761, 2649, 1476, 2928, 2474, 4573, 3627, 6806, 2110, 1751, 3191, 2930, 6609, 3968, 2707, 3101, 2746, 2832, 1320, 6656, 20955, 2300, 1740, 1982, 4693, 2112, 2234, 2467, 3657, 3350, 2074, 8477, 4779, 2427, 1739, 2541, 2968, 3917, 2767, 3098, 1644, 1721, 1785, 8468, 2628, 2398, 1577, 3367, 1816, 2940, 9203, 3235, 2869, 2369, 2905, 4229, 1654, 2058, 3060, 2247, 5651, 3609, 5792, 2590, 10601, 2079, 2002, 3804, 3015, 5210, 3319, 2778, 1614, 2424, 2121, 3114, 4375, 2777, 3892, 3904, 1671, 4446, 2921, 3301, 3054, 2430, 4663, 2113, 3207, 2189, 3369, 2169, 1708, 3388, 2949, 2620, 3192, 2656, 3034, 2162, 4203, 3256, 1481, 3977, 3877, 3908, 2221, 2751, 6160, 3273, 3849, 4009, 4468, 9304, 3784, 2511, 3472, 3990, 3498, 2775, 1715, 2694, 1737, 2441, 1931, 1524, 1429, 2677, 7314, 2184, 3426, 2742, 2649, 4836, 4526, 3583, 2640, 3570, 2858, 3324, 2446, 1822, 6836, 2077, 2119, 2484, 2847, 4063, 3638, 5552, 7376, 1392, 4446, 2403, 4427, 1701, 3225, 3371, 4216, 4064, 2023, 1937, 2111, 2645, 4374, 4590, 2261, 4618, 1518, 3361, 2333, 1721, 3012, 3490, 4952, 4141, 3904, 2335, 3469, 1310, 1456, 2537, 3454, 3856, 3202, 1794, 2515, 11330, 1528, 7000, 3033, 2870, 2194, 2629, 2438, 2250, 3158, 3526, 4443, 7126, 2559, 2404, 4822, 5383, 2477, 6797, 2486, 3611, 2887, 2588, 7121, 2383, 1996, 3107, 1836, 2877, 3455, 2662, 3365, 1432, 2911, 2298, 3639, 2781, 2950, 5031, 1657, 2398, 3564, 3139, 2453, 6067, 4692, 2988, 1360, 2856, 3357, 4918, 2456, 2795, 4763, 2880, 4348, 2375, 3168, 3078, 2798, 2563, 2208, 3349, 2020, 2492, 3925, 2368, 3459, 3118, 3132, 1390, 3540, 2179, 5210, 3135, 1867, 3557, 5771, 2317, 4784, 6924, 12627, 3089, 3870, 9835, 1528, 2463, 3077, 3224, 3195, 5640, 1519, 5676, 1831, 3757, 1775, 2624, 1187, 2453, 2530, 11868, 1785, 2721, 5088, 2440, 3053, 3169, 2694, 6848, 3744, 3735, 3782, 3095, 4393, 4355, 1358, 2326, 3358, 3809, 2166, 2365, 1948, 2432, 1454, 4007, 4098, 1719, 2632, 2534, 5321, 3131, 3328, 2018, 1942, 2079, 5154, 1860, 3634, 1793, 3701, 6359, 1529, 2043, 2141, 1824, 7025, 2181, 4234, 2506, 2492, 2303, 3824, 9920, 3020, 1942, 4101, 1696, 6960, 3474, 2060, 3708, 3019, 1604, 2245, 4782, 2806, 2001, 3413, 2197, 6061, 1903, 2718, 1495, 2639, 7109, 2448, 3466, 7056, 5322, 2468, 1666, 2485, 2958, 2861, 4029, 2802, 1627, 2102, 2876, 1691, 3267, 2146, 3896, 2177, 5096, 3122, 2033, 1983, 2356, 2615, 3077, 2230, 2595, 1859, 2541, 1841, 2307, 3645, 1357, 2578, 2795, 2055, 6249, 2730, 2416, 2159, 1551, 3971, 2195, 2594, 1530, 2798, 1936, 1986, 3550, 3026, 3840, 1811, 2704, 2827, 2126, 3863, 2688, 3362, 2663, 3274, 1734, 2359, 3041, 2848, 1480, 3037, 3509, 2654, 2649, 2942, 2240, 2070, 3869, 1817, 3218, 2158, 3286, 1264, 2820, 2895, 3635, 9012, 4725, 8106, 2711, 1949, 3733, 2205, 3942, 2282, 2048, 2515, 3166, 4047, 6384, 2271, 2171, 7943, 5788, 4878, 3052, 2250, 1723, 2917, 3204, 1779, 1892, 5912, 2197, 2650, 3774, 1449, 2455, 2737, 3318, 1787, 6099, 4433, 3953, 5967, 3581, 1708, 3245, 4737, 2837, 2139, 3303, 11441, 2863, 2847, 2503, 2059, 5244, 3565, 3519, 2773, 3402, 1755, 4801, 2227, 3554, 1827, 3394, 1988, 3058, 2439, 2011, 5135, 4376, 3269, 4452, 3834, 6948, 3196, 1431, 2399, 2955, 2902, 5713, 3248, 1647, 2230, 5084, 3190, 5345, 3433, 4356, 3086, 4280, 1691, 2795, 3692, 2066, 3412, 1800, 13574, 1882, 2682, 2410, 1719, 3496, 2888, 5636, 2682, 2934, 1584, 2233, 2320, 1683, 4665, 1601, 3314, 2435, 2146, 1943, 2053, 3518, 3187, 2478, 2250, 1727, 1774, 2202, 2230, 4264, 3685, 3860, 2934, 2638, 2199, 3741, 4248, 1926, 2440, 2594, 2317, 3061, 4874, 4351, 3459, 1958, 1757, 1364, 6777, 2112, 6554, 2341, 3285, 5393, 2119, 2842, 2450, 2129, 1811, 2011, 2834, 3713, 2391, 2624, 5969, 1803, 1827, 2319, 2741, 3752, 3010, 1864, 1673, 4545, 2566, 1643, 2551, 2022, 3219, 1451, 3257, 1876, 2503, 1959, 2623, 2765, 4129, 1945, 4771, 2667, 2616, 1906, 3178, 4555, 3050, 3392, 2675, 3015, 2945, 2490, 3648, 1858, 4479, 1107, 2277, 3984, 3516, 3027, 3195, 3010, 2073, 4041, 4315, 4044, 2503, 4325, 6082, 2571, 2312, 4721, 1486, 3372, 2869, 2852, 1796, 1840, 2534, 3634, 2358, 2401, 2192, 4237, 4754, 1709, 2343, 2394, 3514, 2723, 3260, 1925, 5167, 2444, 2082, 3368, 2266, 6175, 2825, 2773, 2892, 2566, 1958, 2860, 1735, 2166, 2236, 3726, 1756, 5488, 3641, 5956, 4145, 2339, 3449, 4151, 5309, 1557, 3622, 1810, 1735, 2259, 2496, 2582, 3006, 4549, 2818, 2082, 1873, 2440, 3492, 2252, 4573, 5642, 1553, 2212, 3191, 2345, 3009, 2065, 3995, 9688, 2558, 2585, 5603, 4295, 1812, 3173, 2423, 4785, 2423, 2232, 4470, 14271, 2331, 3398, 2133, 3399, 4000, 4614, 2329, 1572, 1695, 3269, 2044, 1651, 1437, 4001, 6619, 1655, 2871, 6315, 2015, 2185, 2297, 2545, 2950, 2011, 3311, 7175, 2457, 1308, 2019, 2661, 2611, 2798, 3036, 1750, 2895, 1959, 5295, 1856, 1955, 2725, 1764, 2584, 3135, 6833, 2855, 2626, 2808, 2632, 2794, 2877, 5219, 6132, 4428, 3989, 1449, 2798, 2719, 4724, 2901, 3214, 1776, 2482, 2131, 3033, 2025, 4261, 9460, 7742, 2868, 3021, 4049, 3167, 1996, 2873, 2514, 2031, 2408, 9691, 4124, 4306, 4061, 1680, 4858, 2032, 3619, 1767, 2877, 5775, 2302, 3420, 1683, 3809, 2790, 5798, 5760, 2639, 2595, 6307, 2665, 2804, 7715, 2969, 1676, 2722, 2407, 1984, 2664, 2047, 3273, 3855, 3421, 2548, 1961, 2366, 2271, 1789, 2202, 5773, 2192, 2898, 1301, 2383, 2666, 2382, 2213, 5010, 2951, 1964, 2341, 4005, 2444, 2062, 4002, 3017, 2138, 2552, 2325, 2961, 2237, 3311, 2705, 2373, 2292, 2496, 1606, 2965, 2222, 1853, 3281, 4489, 3497, 1705, 4260, 2459, 2442, 2707, 3495, 1945, 2553, 6004, 3767, 3990, 3291, 3209, 2222, 3570, 2373, 3911, 4804, 2230, 5802, 2117, 2359, 4972, 2330, 2232, 3354, 2300, 5387, 2376, 2796, 2226, 2554, 1739, 5245, 2470, 2147, 2643, 2961, 2588, 2695, 4627, 1839, 1933, 2233, 3133, 4399, 2756, 2381, 2870, 2236, 8761, 3426, 1976, 3668, 2323, 2223, 5181, 1915, 2659, 4247, 3455, 4464, 3387, 4578, 5071, 1154, 2365, 4993, 2213, 1776, 2318, 21399, 3106, 3334, 5426, 1951, 4227, 2548, 3279, 10281, 3201, 3190, 2874, 2625, 3361, 3825, 3226, 2617, 3361, 1398, 3674, 2105, 6827, 1927, 2621, 2281, 2364, 3566, 3746, 3017, 2755, 2298, 2480, 1938, 2845, 2237, 2581, 1753, 2050, 2883, 1702, 4789, 2711, 3536, 1942, 2665, 2631, 2184, 2875, 2700, 5504, 2338, 3554, 3595, 3596, 4367, 2883, 2197, 3964, 2565, 2560, 1703, 2491, 1408, 4490, 7309, 3064, 1294, 2842, 2257, 2582, 1352, 4485, 4303, 2807, 5524, 1707, 2057, 2057, 1853, 3348, 2616, 2327, 1754, 5924, 1797, 1395, 2439, 2680, 3250, 1779, 2719, 1558, 4469, 1858, 1965, 3109, 2930, 4197, 1793, 1939, 2341, 1950, 1921, 3495, 2943, 3416, 2056, 3580, 1839, 2150, 2893, 2534, 3282, 6491, 7718, 4058, 8312, 3899, 2325, 3852, 2021, 2455, 2171, 2834, 3632, 4409, 2699, 2625, 3465, 2751, 3092, 2391, 3409, 3669, 1348, 2861, 1879, 2284, 3014, 2341, 2088, 2004, 2708, 2524, 1680, 2562, 1936, 2308, 1845, 5170, 2287, 2746, 1792, 2380, 2314, 3953, 2194, 5035, 2754, 2917, 3349, 1993, 4153, 2925, 4011, 2480, 2704, 2326, 1594, 4390, 2194, 2826, 4798, 4132, 2492, 3084, 2556, 2570, 1616, 2177, 3959, 3387, 5239, 2585, 2455, 3685, 2325, 2551, 2769, 2787, 2109, 3274, 2162, 2134, 2520, 1977, 2357, 1375, 3022, 3690, 2473, 2961, 2611, 11821, 3372, 1768, 5266, 2329, 2344, 1796, 1742, 2359, 2459, 1586, 3709, 2131, 2554, 1564, 2215, 5798, 3113, 3041, 3463, 2566, 3219, 2330, 3997, 2403, 1911, 2423, 1798, 1833, 3527, 2932, 2534, 1631, 2066, 4538, 3189, 2549, 1733, 6669, 5731, 2390, 3100, 7548, 2880, 2396, 3654, 4783, 6170, 4889, 3163, 3772, 1750, 1486, 1633, 1999, 1658, 1730, 1502, 3600, 1894, 5575, 3316, 4022, 4024, 2535, 3013, 5397, 3092, 5507, 4655, 2061, 2209, 2336, 3676, 1365, 13645, 2292, 6711, 3858, 3933, 1548, 2424, 2369, 3058, 1423, 2481, 1242, 7126, 1693, 3089, 2809, 3740, 2454, 3395, 1606, 4418, 3612, 3271, 1609, 3222, 2257, 3143, 4289, 4357, 2151, 3206, 2806, 3790, 3592, 5046, 2804, 1396, 2825, 1879, 4655, 1820, 5494, 1895, 4179, 3827, 3521, 2110, 5806, 4631, 2313, 5316, 2388, 2659, 2208, 2084, 2592, 1743, 2943, 3812, 1841, 2539, 2871, 3595, 3123, 1754, 2488, 3644, 5192, 2347, 1763, 8916, 2378, 1768, 4566, 3227, 4186, 3932, 2907, 1824, 2330, 4367, 4432, 3617, 2804, 2210, 1932, 3638, 3485, 2810, 5985, 2776, 2107, 1985, 3091, 3170, 3792, 1297, 5002, 2535, 1975, 2753, 2621, 6044, 1646, 2099, 2197, 3990, 3129, 3616, 2357, 2042, 3607, 3822, 3631, 2962, 3618, 3684, 4743, 2524, 5788, 1543, 3525, 3905, 3114, 1827, 2271, 5206, 5759, 2156, 1629, 2165, 1709, 3199, 2202, 2048, 1908, 7543, 2061, 3094, 2440, 2867, 2533, 3949, 2340, 2551, 1787, 2228, 4098, 7707, 2264, 2991, 2226, 1894, 2044, 2659, 6406, 1506, 2381, 3518, 1536, 3089, 3322, 2923, 1922, 10127, 3101, 2352, 4332, 1754, 2462, 2544, 3138, 2190, 3658, 1369, 4191, 2605, 2953, 4180, 2804, 2495, 2826, 3869, 4277, 3945, 2390, 3135, 4000, 2114, 4937, 2854, 2595, 3120, 5304, 5013, 1902, 4624, 2725, 1726, 4216, 2928, 1818, 3928, 5024, 1838, 3137, 2440, 2942, 7077, 4611, 2556, 2059, 2268, 6067, 1841, 3232, 3159, 3493, 1978, 1541, 1871, 3011, 4799, 1689, 2486, 2752, 3147, 2958, 1669, 1461, 4761, 4818, 3115, 2677, 2223, 2455, 3059, 2114, 1147, 2274, 2217, 1629, 3721, 1413, 2351, 2725, 3021, 3941, 2166, 6070, 3067, 2799, 2229, 2589, 1808, 5666, 2399, 3913, 2401, 4092, 3737, 3883, 2764, 3198, 1615, 4617, 5911, 3673, 1927, 1480, 2035, 3710, 2716, 1712, 2623, 1984, 2512, 3954, 3243, 2952, 2634, 4050, 1821, 6298, 2435, 7740, 2536, 2681, 2221, 1584, 2983, 4971, 3061, 2135, 1853, 11582, 2868, 4346, 2334, 3206, 4000, 1855, 2380, 3208, 2524, 3354, 2794, 3058, 2951, 3633, 1830, 2088, 3136, 2156, 3513, 4345, 4315, 1733, 4808, 3830, 4229, 2703, 3492, 2530, 1904, 2620, 3131, 1736, 4492, 2715, 6275, 6817, 3603, 4497, 2374, 3541, 4219, 2090, 7292, 2024, 2227, 1833, 1826, 2270, 2604, 3659, 2730, 3476, 3993, 3453, 3693, 2586, 5175, 2596, 2210, 2700, 4319, 2766, 2282, 12637, 3534, 2373, 2477, 2209, 3096, 1677, 2332, 2671, 5844, 2930, 1844, 2673, 7856, 7781, 2809, 2542, 5402, 3113, 3822, 2523, 3598, 1481, 1631, 2400, 1940, 2292, 4025, 2321, 3298, 1699, 2794, 2736, 2949, 2927, 1527, 4444, 4414, 1433, 1909, 2558, 2728, 5360, 2093, 2175, 3391, 3060, 1820, 1829, 4730, 1980, 2726, 3658, 4707, 2678, 4968, 2174, 2423, 3174, 4193, 2165, 2493, 3485, 2876, 8590, 3627, 1825, 2028, 4501, 5719, 3030, 4104, 9048, 3065, 2587, 2591, 7242, 3214, 2872, 3008, 4941, 2527, 3113, 2704, 1805, 3850, 2391, 1456, 3740, 1774, 2451, 2963, 3298, 2173, 4438, 3393, 5453, 1862, 3390, 3279, 5097, 3488, 4368, 3364, 2122, 3619, 2625, 5245, 2642, 3681, 1978, 2886, 3284, 3816, 3459, 3130, 4803, 4180, 5069, 2836, 2602, 1691, 3818, 2396, 1909, 1372, 3565, 2549, 5583, 3594, 2034, 1795, 4548, 3552, 5306, 5684, 2856, 4124, 2814, 2475, 2028, 4421, 1955, 3138, 1586, 2723, 4697, 2163, 1608, 2354, 2843, 2676, 1825, 3321, 3914, 3078, 2961, 2844, 4636, 2487, 2720, 3637, 2869]
# print (f'overall num of pkts={sum(equinixFlowSizes)}')    
if __name__ == '__main__':
    # main (mode='NiceBuckets', runShortSim=True)
    main (mode='IceBuckets', runShortSim=False)
