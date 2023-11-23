import math, random, os, pickle, mmh3
import PclFileParser, settings
import Morris, F2P, CEDAR, RealCntr
import numpy as np
from printf import printf

def remove_existing_files():
    # Removes the specified files if they exist.
    if os.path.exists('../res/pcl_files/RdRmse.pcl'):
        os.remove('../res/pcl_files/RdRmse.pcl')
    if os.path.exists('../res/RdRMSE.res'):
        os.remove('../res/RdRMSE.res')

class CountMinSketch:
    def __init__(self,
                 width,         # the number of counters per row.
                 depth,         # the number of rows or hash functions.
                 num_flows,     #the total number of flows to be estimated.
                 mode,          # It is one of the counter modes.
                 conf,          # it is a dictionary that holds the value of cntrSize, cntrMaxVal, hyperSize and so on.
                 outPutFileName #this a files name which we use with res and pcl file. Eg. outPutFileName.res and outPutFileName.pcl
                 ):
        """
        pair: a tuple of row and column indices used to compute the corresponding counter index. It is used to compute the index of each counter. Instead of adding row with column which gives us wrong mapping, i have
        paired the depth number with the hash value and i have inserted them to the pair list to use the index of the pair as a counter index.
         """
        self.mode, self.width, self.depth, self.num_flows = mode, width, depth, num_flows
        self.numCntrs        =self.width*self.depth
        # Access the values within the conf dictionary using keys
        self.cntrSize       =conf['cntrSize']
        self.cntrMaxVal     =conf['cntrMaxVal']
        self.hyperSize      =conf['hyperSize']
        self.hyperMaxSize   =conf['hyperMaxSize']
        self.outPutFileName =outPutFileName
        self.verbose        =[5, 8]
        self.pair           = [(i, j) for i in range(depth) for j in range(width)]
        if self.mode=='F2P':
            self.countersArray = F2P.CntrMaster(cntrSize=self.cntrSize, hyperSize=self.hyperSize, hyperMaxSize=self.hyperMaxSize, mode='F2P', numCntrs=self.numCntrs, verbose=[])
        elif self.mode=='Morris':
            self.countersArray = Morris.CntrMaster(cntrSize=self.cntrSize, numCntrs=self.numCntrs, a=None, cntrMaxVal=self.cntrMaxVal,verbose=[], estimateAGivenCntrSize=False)
        elif self.mode=='realCounter':
             self.countersArray=RealCounter.CntrMaster(cntrSize=self.cntrSize, numCntrs=self.numCntrs)
        elif self.mode=='CEDAR':
            self.countersArray = CEDAR.CntrMaster(cntrSize=self.cntrSize, delta=None, numCntrs=self.numCntrs, verbose=[], cntrMaxVal=self.cntrMaxVal)  # Initialize the CEDAR counter
        else:
            print(f'Sorry, the mode {self.mode} that you requested is not supported')

    def incNQueryFlow(self, flow):
        """
        When a flow arrives, it is hashed using the hash functions, and the corresponding counters are incremented.
        At the end,  the minimum value of the corresponding counters is turned as the estimate.
        """
        cntrValAfterInc = [0]*self.depth
        for mappedCntr in range(self.depth):
            counterIndex               = self.pair.index((mappedCntr, mmh3.hash(str(flow), seed=mappedCntr) % self.width))
            cntrValAfterInc[mappedCntr]=self.countersArray.incCntr(cntrIdx=counterIndex, factor=1, mult=False, verbose=[])['val']
        return min(cntrValAfterInc)

    def incFlow(self, flow):
        # increment the mapped counters values
        for seed in range(self.depth):
                counterIndex = self.pair.index((seed,mmh3.hash(str(flow), seed) % self.width))
                self.countersArray.incCntr(cntrIdx=counterIndex, factor=int(1), mult=False, verbose=[])

    def queryFlow(self, flow):
        # Query the minimum Morris, CEDAR, real counter value for the given flow by hashing and finding the minimum value among the appropriate counters
        minNum = math.inf
        for seed in range(self.depth):
            counterIndex = self.pair.index((seed,mmh3.hash(str(flow), seed) % self.width))
            self.countersArray.queryCntr(counterIndex)
            minNum       = min(self.countersArray.queryCntr(counterIndex)['val'], minNum)
        return minNum

    def calculateNormalizedRMSE(self):
        """
        This simulation stores the real frequency of a flow at the index of a flow and it queries the estimated value of the flow
        from the array counter after incrementing it using the methodology of count min sketch.
        It calculates the Read Root Mean Square Error (RMSE), Normalized RMSE, normalized RMSE average with its confidence interval
        for different counter modes.At the end, it writes or prints the output to res and pcl files as a dictionary.
        """
        numOfIncrements  = 1000
        numOfExps        =50
        sumOfAllErors    = [0] * numOfExps
        numOfPoints      = [0] * numOfExps # self.numOfPoints[j] will hold the number of points collected for statistic at experiment j.
        for expNum in range(numOfExps):
            realCntr      = np.zeros(self.num_flows)
            sampleProb   = 1
            self.countersArray.rstAllCntrs()  # To reset the value of all counters
            # Increment the counters randomly and update the real values
            for incNum in range(numOfIncrements):
                flow                 = np.random.randint(self.num_flows)
                if self.queryFlow(flow) < self.cntrMaxVal:
                    # Choose a random flow to increment
                    realCntr[flow] += 1  # increment the real counter value of the flow upon its arrival
                    if sampleProb==1 and random.random() < sampleProb:
                        sumOfAllErors[expNum]    +=((realCntr[flow] - self.incNQueryFlow(flow))/realCntr[flow])**2
                        numOfPoints[expNum]      += 1
        # Compute the Root Mean Square Error (RMSE) and Normalized RMSE over all experiments using the sumOfAllErors
        RMSE                 = [math.sqrt(sumOfAllErors[expNum]/numOfPoints[expNum]) for expNum in range(numOfExps)]
        Normalized_RMSE      = [RMSE[expNum]/numOfPoints[expNum] for expNum in range(numOfExps)]
        log_file = open(f'../res/log_files/{self.outPutFileName}.log', 'w')
        if (settings.VERBOSE_LOG in self.verbose):
            printf (log_file, 'Normalized_RMSE=\n{0}\n'.format(Normalized_RMSE))
        normRmseAvg          = np.average(Normalized_RMSE)
        normRmseConfInterval = settings.confInterval(ar=Normalized_RMSE, avg=normRmseAvg)
        # Write the results to a file and return them as a dictionary
        dict                 = {
                                'mode'    :self.mode,
                                'numCntrs': self.numCntrs,
                                'Avg'     : normRmseAvg,
                                'Lo'      : normRmseConfInterval[0],
                                'Hi'      : normRmseConfInterval[1]
                              }
        self.dumpDictToPcl(dict)      # this takes the dict dictionary as a parameter to dump it to .pcl file
        self.writeDictToResFile(dict) # this takes the dict dictionary as a parameter to print it to .res file
    def dumpDictToPcl (self, dict):
        """
        Dump a single dict of data into pclFile
        """
        with open(f'../res/pcl_files/{self.outPutFileName}.pcl', 'ab') as f:
            pickle.dump(dict, f)

    def writeDictToResFile (self, dict):
        """
        Write a single dict of data into resFile
        """
        resFile = open (f'../res/{self.outPutFileName}.res', 'a+')
        printf (resFile, f'{dict}\n\n')

def main():
    """
    This iterates over different configurations, counter modes and widths with a fixed number of depth. the configuration which is a
     list of dictionaries, which found in the settings.py holds different conter sizes, counter max value, hyper sizes and so on.
     The counter modes are specified as 'F2P', 'CEDAR', 'Morris', and 'realCounter'. The depth of the array counter is set to 8.
     The counter sizes are specified as [8, 10, 12], and the widths are generated using the range function with a step size of 11.
     This initializes the count min sketch variables and calls the calculateNormalizedRMSE function
    """
    remove_existing_files()  # Remove existing files
    counter_modes = ['F2P', 'CEDAR', 'Morris','realCounter']  # List of counter modes
    num_flows     = 100  # Number of flows
    depth         = 2 # Depth of the array counter
    cntrSizes     = [6, 8, 10, 11, 12, 13, 14, 16]  # Counter sizes
    for conf in settings.Confs: # Iterate over each dictionary in settings.Confs list
        # Check if the 'cntrSize' the conf dictionary is in the specified counter sizes
        if conf['cntrSize'] in cntrSizes:
            for counter_mode in counter_modes:
                for width in range(2,num_flows//2, 10):
                    cmc = CountMinSketch(width=width, depth=depth, num_flows=num_flows, mode=counter_mode, conf=conf, outPutFileName='RdNMSE_{}depth_{}bits'.format(depth, conf['cntrSize']))
                    cmc.calculateNormalizedRMSE()
            # Create a PclFileParser object
            parser = PclFileParser.PclFileParser()
            # Read in data from a PCL file
            parser.rdPcl(pclFileName='RdNMSE_{}depth_{}bits'.format(depth, conf['cntrSize']))
            # Generate a plot showing normRmseAvg versus number of counter for each counter modes.
            parser.NRMSEVsWidthPlot(modes=counter_modes, pclFileName='RdNMSE_{}depth_{}bits'.format(depth, conf['cntrSize']))

if __name__ == '__main__':
    # Call the main function
    main()
