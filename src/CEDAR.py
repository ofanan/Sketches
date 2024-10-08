# This class implements the CEDAR counter.
# CEDAR is described in the paper: "Estimators also need shared values to grow together", Tsidon, Erez and Hanniel, Iddo and Keslassy, Isaac, Infocom'15.
import random, math, numpy as np

from printf import printf
import settings, Cntr
from settings import *
np.set_printoptions(precision=1)

# The 'delta' parameter determines CEDAR's accuracy.
# Given a counter size and maximum value to count, the function findMinDeltaByMaxVal finds the minimal delta. using binary search.
# To prevent overflows, the search range for some parameters should be limited corresponding to the counter's size. 
# This is done using the list of dicts below.
preComputedData = [
                 {'cntrSize' : 4,    'deltaLo' : 0.0001,    'deltaHi' : 0.3},
                 {'cntrSize' : 5,    'deltaLo' : 0.0001,    'deltaHi' : 0.3},
                 {'cntrSize' : 6,    'deltaLo' : 0.0001,    'deltaHi' : 0.3},
                 {'cntrSize' : 7,    'deltaLo' : 0.0001,    'deltaHi' : 0.3},
                 {'cntrSize' : 8,    'deltaLo' : 0.00001,   'deltaHi' : 0.2},
                 {'cntrSize' : 9,    'deltaLo' : 0.0001,    'deltaHi' : 0.2},
                 {'cntrSize' : 10,   'deltaLo' : 0.0001,    'deltaHi' : 0.2},
                 {'cntrSize' : 11,   'deltaLo' : 0.0001,    'deltaHi' : 0.15},
                 {'cntrSize' : 12,   'deltaLo' : 0.0001,    'deltaHi' : 0.13},
                 {'cntrSize' : 13,   'deltaLo' : 0.0001,    'deltaHi' : 0.1},
                 {'cntrSize' : 14,   'deltaLo' : 0.00001,   'deltaHi' : 0.1},
                 {'cntrSize' : 15,   'deltaLo' : 0.00001,   'deltaHi' : 0.08},
                 {'cntrSize' : 16,   'deltaLo' : 0.00001,   'deltaHi' : 0.07},
                 ]

class CntrMaster (Cntr.CntrMaster):
    """
    Generate, check and parse counters
    """
    # Generates a string that details the counter's settings (param vals).
    genSettingsStr = lambda self : 'Cedar_n{}_d{:.6f}'.format(self.cntrSize, self.delta)
    
    # Calculate the diff of an estimator from the previous estimator, given the sum of the previous diffs
    calcDiff = lambda self, sum_of_prev_diffs: (1 + 2 * self.delta ** 2 * sum_of_prev_diffs) / (1 - self.delta ** 2)

    # print the details of the counter in a convenient way
    printCntrLine = lambda self, cntrSize, delta, numCntrs, mantVal, cntrVal: print('cntrSize={}, delta={}' .format(cntrSize, delta))

   
    # Given the cntr's index, returns estimation
    estimatedValOfCntrIdx = lambda self, idx : self.estimators[self.cntrs[idx]]

    def __init__(self, 
            cntrSize       = 8, # num of bits in each counter.
            delta          = None, # Delta - the max relative error, as detailed in the paper CEDAR. 
            numCntrs       = 1, # number of counters in the array.
            cntrMaxVal     = None, # Max value to be reached by a counter. When Delta==None, the initiator uses this value, and calculates (using binary search) the minimum delta that allows reaching this maximum value.
            verbose        = [], 
        ):
        """
        Initialize an array of cntrSize counters. The cntrs are initialized to 0.
        """
        super(CntrMaster, self).__init__ (
            cntrSize    = cntrSize, 
            numCntrs    = numCntrs, 
            verbose     = verbose)
        self.cntrMaxVal    = cntrMaxVal
        self.numEstimators = 2**self.cntrSize
        self.rst () # reset all the counters
        
        if (delta==None):           
                if (cntrMaxVal==None):
                    print ('error: the input arguments should include either delta or cntrMaxVal')
                    exit ()
                
                # Delta is not given, but cntrMaxVal is given --> calculate the optimal delta for this cntrSize and cntrMaxVal
                self.cntrMaxVal = cntrMaxVal
                if (settings.VERBOSE_DETAILS in self.verbose):
                    self.detailFile = open ('../log/CEDAR_details.log', 'w')
                self.findMinDeltaByMaxVal(targetMaxVal=self.cntrMaxVal)
                if (settings.VERBOSE_DETAILS in self.verbose):
                    printf (self.detailFile, 'cntrSize={}, cntrMaxVal={}, found delta={}\n' .format (self.cntrSize, self.cntrMaxVal, self.delta))
                    for i in range(len(self.estimators)):
                        printf (self.detailFile, 'sharedEstimator[{}]={:.4f}\n' .format(i, self.estimators[i]))
                
        else:
            self.delta         = delta
            self.calcDiffsNSharedEstimators ()
        
    

    def cntr2num (self, cntr): 
        """
        # Given the counter (as a binary vector string) return the value it represents
        Given a cntr, return the value it represents
        """
        if isinstance (cntr, str):
            cntr = int(cntr, base=2)
        return self.estimators[cntr] 

    def findPreComputedDatum (self):
        """
        Returns the precomputed datum with the requested cntrSize.
        """
        preComputedDatum = [item for item in preComputedData if item['cntrSize']==self.cntrSize]
        if len(preComputedDatum)==0:
            settings.error ('Sorry, but the requested cntrSize {self.cntrSize} is currently not supported by CEDAR')
        elif len(preComputedDatum)>1:
            settings.error ('More then one entry in preComputedData for the requested cntrSize {self.cntrSize}')
        return preComputedDatum[0]
               

    def rst (self):
        """
        Reset all the counters.
        """
        if self.cntrSize <= 8: 
            self.cntrs = np.zeros (self.numCntrs, 'uint8')
        elif self.cntrSize <= 16: 
            self.cntrs = np.zeros (self.numCntrs, 'uint16')
        elif self.cntrSize <= 32: 
            self.cntrs = np.zeros (self.numCntrs, 'uint32')
        else:
            settings.error ('in CEDAR.rst() : sorry, cntrSize>32 is not supported yet.')

        
    def calcDiffsNSharedEstimators (
            self,
            delta = None # CEDAR's delta parameter, defining the accuracy. When None, use self.delta
            ):
        """
        Calculate the values of the shared estimators and the diffs between them based on the delta accuracy parameter, as detailed in the paper CEDAR.
        """
        self.estimators = np.zeros (self.numEstimators)
        self.diffs      = np.zeros (self.numEstimators-1) 
        for i in range (1, self.numEstimators):
            self.diffs[i-1] = self.calcDiff(self.estimators[i-1])
            self.estimators[i] = self.estimators[i-1] + self.diffs[i-1] 
        self.cntrMaxVal = self.estimators[-1]    
    
    def rstCntr (self, cntrIdx=0):
        """
        """
        self.cntrs[cntrIdx] = 0

        
    def incCntr(self, cntrIdx=0, factor=1, mult=False, verbose=[]):
        """
        Increase a counter by a given factor.
        Input:
        cntrIdx - index of the cntr to increment, in the array of cntrs.
        mult - if true, multiply the counter by factor. Else, increase the counter by factor.
        factor - the additive/multiplicative coefficient.
        verbose - determines which data will be written to the screen.
        Output:
        cntrDict: a dictionary representing the modified counter where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.
        Operation:
        Define cntrVal as the current counter's value. 
        Then, targetValue = cntrVal*factor (if mult==True), and targetValue = cntrVal + factor (otherwise).  
        If targetValue > maximum cntr's value, return the a cntr representing the max possible value. 
        If targetValue < 0, return a cntr representing 0.
        If targetValue can be represented correctly by the counter, return the exact representation.
        Else, use probabilistic cntr's modification.
        
        If verbose==settings.VERBOSE_DETAILS, the function will print to stdout:
        - the target value (the cntr's current value + factor)
          - cntrDict['cntrVec'] - the binary counter.
          - cntrDict['val']  - the counter's value.
        """
        settings.checkCntrIdx(cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='CEDAR')
        for i in range(factor):
            # The probability to increment is calculated  according to the diff
            if (self.cntrs[cntrIdx] == self.numEstimators-1): # reached the largest estimator --> cannot further inc
                if (settings.VERBOSE_NOTE in self.verbose):
                    print ('note: tried to inc cntr {} above the maximal estimator value of {}' .format (cntrIdx, self.estimators[-1]))
                break
            probOfFurtherInc = 1/self.diffs[self.cntrs[cntrIdx]]
            if random.random() < probOfFurtherInc:
                if (settings.VERBOSE_DETAILS in verbose): 
                    print ('oldVal={:.0f}, incedVal={:.0f}, probOfFurtherInc={:.6f}'
                            .format (self.estimators[self.cntrs[cntrIdx]], self.estimators[self.cntrs[cntrIdx]+1], probOfFurtherInc))
                self.cntrs[cntrIdx] += 1

        return {'cntrVec': np.binary_repr(self.cntrs[cntrIdx], self.cntrSize), 'val': self.estimators[self.cntrs[cntrIdx]]}

    def incCntrBy1GetVal (self, cntrIdx=0):
        """
        """
        if (self.cntrs[cntrIdx] == self.numEstimators-1): # reached the largest estimator --> cannot further inc
            return self.estimators[self.cntrs[cntrIdx]]
        if random.random() < 1/self.diffs[self.cntrs[cntrIdx]]:
            self.cntrs[cntrIdx] += 1
        return self.estimators[self.cntrs[cntrIdx]]

    def queryCntr (self, 
            cntrIdx  = 0, #  
            getVal   = True # If True, return only the counter's value. Else, return cntrDic - a dictionary, where cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.
        ):
        """
        Query a cntr.
        Input: 
         
        Output:
        cntrDic: a dictionary, where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.        
        """
        if getVal:
            return self.estimators[self.cntrs[cntrIdx]]
        settings.checkCntrIdx(cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='CEDAR')
        return {'cntrVec': np.binary_repr(self.cntrs[cntrIdx], self.cntrSize), 'val': self.estimators[self.cntrs[cntrIdx]]}

    def findMinDeltaByMaxVal (
            self,
            targetMaxVal, # Value that the counter must reach
            resolution = 0.0000001, # minimal resolution (stopping condition) at the binary search
            deltaLo     = None,     # lowest value of the delta to consider at the binary search. When None, use pre-computed values.             
            deltaHi     = None,     # highest value of the delta to consider at the binary search. When None, use pre-computed values.             
            ):
        """
        Given a target maximum countable value, return the minimal 'delta' parameter that reaches this value, 
        for the current counter's size.
        delta value determines the expected error: a higher delta implies a higher estimated error.
        The min necessary delta is found through a binary search.
        Inputs:   
        * deltaLo - initial lower val for the binary search
        * deltaHi - initial higher val for the binary search
        * resolution = minimum difference (deltaHi-deltaLo); when reached - break the binary search.
        """

        preComputedDatum = self.findPreComputedDatum ()
        if deltaLo==None:
            deltaLo, deltaHi = preComputedDatum['deltaLo'], preComputedDatum['deltaHi']

        # check first the extreme cases
        self.delta = deltaHi
        self.calcDiffsNSharedEstimators ()
        if (self.cntrMaxVal < targetMaxVal):
            error (f'cannot reach maxVal={targetMaxVal} even with highest delta, deltaHi={deltaHi}. Skipping binary search') #$$
            return

        while (True):
            if (deltaHi - deltaLo < resolution): # converged. Still, need to check whether this delta is high enough.
                self.calcDiffsNSharedEstimators ()
                if (self.cntrMaxVal >= targetMaxVal): # can reach maxVal with this delta --> Good
                    return
                # now we know that cannot reach targetMaxVal with the current delta
                self.delta += resolution
                self.calcDiffsNSharedEstimators ()
                if (self.cntrMaxVal < targetMaxVal): 
                    print ('problem at binary search')
                    exit ()
                return
                
            self.delta = (deltaLo + deltaHi)/2
            if (settings.VERBOSE_DETAILS in self.verbose):
                printf (self.detailFile, 'delta={}\n' .format (self.delta))
            self.calcDiffsNSharedEstimators ()
            if (self.cntrMaxVal==targetMaxVal): # found exact match 
                break
            if (self.cntrMaxVal < targetMaxVal): # can't reach maxVal with this delta --> need larger delta value
                deltaLo = self.delta
            else: # maxVal > targetMaxVal --> reached the maximum value - try to decrease delta, to find a tighter value.
                deltaHi = self.delta
        return self.delta             

    def upScale (self):
        """
        Allow down-sampling:
        - Calculate a new "delta" parameter that allows reaching a higher cntrMaxVal.
        - Calculate new cntrs' value to keep roughly the estimation as before the upscale.  
        """
        prevEstimators = self.estimators.copy()
        # prevDelta, prevDiffs, prevEstimators = self.delta, self.diffs.copy(), self.estimators.copy()
        prevCntrMaxVal   = self.cntrMaxVal 
        self.cntrMaxVal *= 2
        
        self.findMinDeltaByMaxVal (
            targetMaxVal    = self.cntrMaxVal,
            deltaLo         = 0.00001,
            deltaHi         = deltaHi
        )                
                
        if VERBOSE_DEBUG in self.verbose:
            self.cntrs = [i for i in range(self.numCntrs)]
                        
        for cntrIdx in range(self.numCntrs):
            orgVal = prevEstimators[self.cntrs[cntrIdx]]
            newEstIdx = 0
            while self.estimators[newEstIdx] < orgVal:
                newEstIdx += 1
            # Now we know that self.estimators[newEstIdx] >= orgVal
            if self.estimators[newEstIdx]==orgVal:
                self.cntrs[cntrIdx] = newEstIdx
                if VERBOSE_DEBUG in self.verbose:
                    printf (self.logFile, 'orgVal=val={:.1f}\n' .format (orgVal))
                continue
            if random.random() < (orgVal-self.estimators[newEstIdx-1])/(self.estimators[newEstIdx]-self.estimators[newEstIdx-1]):
                self.cntrs[cntrIdx] = newEstIdx
            else:
                self.cntrs[cntrIdx] = newEstIdx-1
            if VERBOSE_DEBUG in self.verbose:
                floorVal = self.estimators[newEstIdx-1]
                ceilVal  = self.estimators[newEstIdx]
                printf (self.logFile, 'orgVal={:.1f}, floorVal={:.1f}, ceilVal={:.1f}, val={:.1f}\n' 
                       .format (orgVal, floorVal, ceilVal, self.estimators[self.cntrs[cntrIdx]]))
        
        if VERBOSE_DEBUG in self.verbose:
            printf (self.logFile, 'Printing all estimators\n')
            for estimator in self.estimators:
                printf (self.logFile, '{:.1f} ' .format(estimator)) 
        
    def printCntrs (self, outputFile=None) -> None:
        """
        Format-print all the counters as a single the array, to the given file.
        """
        if outputFile==None:
            print ('cntrs={} ' .format([self.cntrInt2num(cntr) for cntr in self.cntrs]))
        else:
            for cntr in self.cntrs:
                printf (outputFile, '{:.0f} ' .format(self.cntrInt2num(cntr)))
    
    def printEstimators (self, outputFile=None) -> None:
        """
        Format-print all the counters as a single the array, to the given file.
        """
        if outputFile==None:
            print ('eps={:.3f}, estimators={}' .format (self.epsilon, self.estimators))            
        else:
            for cntr in self.cntrs:
                printf (outputFile, '{:.0f} ' .format(self.cntrInt2num(cntr)))
    

# def printAllVals(cntrSize=8, delta=None, cntrMaxVal=None, verbose=[]):
#     """
#     Loop over all the binary combinations of the given counter size.
#     For each combination, print to file the respective counter, and its value.
#     The prints are sorted in an increasing order of values.
#     """
#     listOfVals = []
#     myCntrMaster = CntrMaster(cntrSize=cntrSize, delta=delta, cntrMaxVal=cntrMaxVal, numCntrs=1)
#     for num in range(2 ** cntrSize):
#         val = myCntrMaster.cntrInt2num(num)
#         listOfVals.append ({'cntrVec' : np.binary_repr(num, cntrSize), 'val' : val})
#
#
#     if settings.VERBOSE_RES in verbose:
#         outputFile = open('../res/single_cntr_log_files/{}.res'.format(myCntrMaster.genSettingsStr()), 'w')
#         for item in listOfVals:
#             printf(outputFile, '{}={:.1f}\n'.format(item['cntrVec'], item['val']))

# \frac{\left(\left(1+2\cdot \:\:x^2\right)^L-1\right)}{2x^2}\left(1+x^2\right)


# myCntrMaster = CntrMaster (
#     numCntrs    = 2**6,
#     cntrSize    = 6, 
#     cntrMaxVal  = 1000,
#     verbose     = [VERBOSE_DEBUG]
# ) 
# logFile = open (f'../res/log_files/{myCntrMaster.genSettingsStr()}.log', 'w')
# myCntrMaster.setLogFile (logFile)
# myCntrMaster.upScale()