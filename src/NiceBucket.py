# This class implements a single NiceBucket.
import random, math, numpy as np
from printf import printf
import settings, IceBucket
from IceBucket import calcEstimatorGivenEpsilon, findPreComputedDatum, calcEstimatorGivenEpsilon
from ctypes.wintypes import BOOLEAN
# To prevent overflows, the search range for some parameters should be limited corresponding to the counter's size. 
# This is done using the list of dicts below.
# The data below is used also to determine the 'epsilonStep' used in ICE_buckets. 

class CntrMaster(IceBucket.CntrMaster):
    """
    Generate, check and parse counters
    """
    # Generates a string that details the counter's settings (param vals).
    genSettingsStr = lambda self : 'Nice_n{}'.format(self.cntrSize)
    
    def __init__(self, 
                 cntrSize           = 8, # num of bits in each counter.
                 numCntrs           = 1, # number of counters in the array.
                 numEpsilonSteps    = None,    # number of different possible estimation scales - a power of two.
                 cntrMaxVal         = None, # Max value to be reached by a counter. 
                 verbose            = [], 
                 id                 = 0,
                 isXlBkt            = False,
                 ):
        """
        Initialize an array of cntrSize counters. The cntrs are initialized to 0.
        """
        self.cntrSize, self.numCntrs, self.cntrMaxVal = cntrSize, numCntrs, cntrMaxVal
        self.id, self.isXlBkt, self.verbose = id, isXlBkt, verbose
        self.numEstimators = 2**self.cntrSize
        self.rst () # reset all the counters
        self.numEpsilonSteps  = numEpsilonSteps 
        self.epsilon          = 0
        self.epsilonStep      = findPreComputedDatum (cntrSize=self.cntrSize)['epsilonStep']
        if not(self.isXlBkt):
            self.isSaturated = [False]*self.numCntrs
    
    def upScale (self):
        """
        Up-scale for reaching a largest maximal value. In particular:
        - Increase the self.epsilon, which determines the error, by self.epsilonStep. Increasing self.epsilon allows reaching larger counted value (at the cost of a larger relative error).
        - calculate the estimators' values using the updated self.epsilon. (localUpscale procedure, defined in [ICE_buckets]).   
        - For each counter ("symbol"), run the "symbol upsclae" procedure, defined in [ICE_buckets].
          This procedure scales-up a single counter after the "epsilon" variable was increased.
        """        
        # Update self.epsilon and then update all the estimators' values accordingly.
        self.prevEpsilon    = self.epsilon  
        self.epsilon       += self.epsilonStep

        # run the localUpscale procedure, defined in [ICE_buckets].
        for cntrIdx in range (self.numCntrs):
            sqEpsilon = self.epsilon**2
            ellTag = math.floor (math.log(1 + (2*sqEpsilon*calcEstimatorGivenEpsilon(self.prevEpsilon, ell=self.cntrs[cntrIdx]))/(1+sqEpsilon))/math.log(1 + 2*sqEpsilon))            
            if random.random() < (calcEstimatorGivenEpsilon(self.prevEpsilon, ell=self.cntrs[cntrIdx]) - calcEstimatorGivenEpsilon(self.epsilon, ellTag))/ (calcEstimatorGivenEpsilon(self.epsilon, ellTag+1) - calcEstimatorGivenEpsilon(self.epsilon, ellTag)):
                self.cntrs[cntrIdx] = ellTag + 1
            else:
                self.cntrs[cntrIdx] = ellTag
        
        
    def incCntrBy1GetVal (self, cntrIdx=0):
        """
        Increase a counter cntrIdx by 1ץ
        Return:
        wasSaturated, valAfterInc, 
        where:
        wasSaturated is True iff the counter was saturated already before incrementing (in this case, no increment is done).
        valAfterInc: if wasSaturated==False, then this is the updated counter's value.
        """
        if not(self.isXlBkt) and self.isSaturated[cntrIdx]:
            return True, None # return values telling that the (regular) counter is saturated
        cntrVal = self.cntrs[cntrIdx]
        if cntrVal==(1 << self.cntrSize) - 1: # reached the largest possible estimated value w/o up-scaling?
            if self.epsilon == ((self.numEpsilonSteps-1) * self.epsilonStep): # already up-scaled as high as possible.
                if self.isXlBkt:
                    settings.error (f'in NiceBucket.incCntrBy1GetVal(). Tried to increment XlBkt {self.id} above the maximum feasible value. cntrSize={self.cntrSize}, numEpsilonSteps={self.numEpsilonSteps}')
                # Now we know that this isn't an XlBkt --> this is a regular bkt
                self.isSaturated[cntrIdx] = True
                if settings.VERBOSE_LOG in self.verbose:
                    printf (self.logFile, f'bkt {self.id} reached max val of regular bkts\n')
                return True, None # return values telling that the (regular) counter is saturated
            if settings.VERBOSE_LOG in self.verbose:
                printf (self.logFile, f'bkt {self.id} is up-scaling. epsilon b4 upscaling={self.epsilon}\n')
            self.upScale () 
            cntrVal = self.cntrs[cntrIdx]# cntrVal is the value in the counter after up-scaling, before incrementing
        curEstimate = calcEstimatorGivenEpsilon(self.epsilon, ell=cntrVal)
        incEstimate = calcEstimatorGivenEpsilon(self.epsilon, ell=cntrVal+1)
        diff        = incEstimate - curEstimate
        if diff==1 or random.random () < 1/diff:  
            self.cntrs[cntrIdx] += 1
            return False, incEstimate
        return False, curEstimate

    def incCntr(self, cntrIdx=0, factor=1, mult=False):
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
        print ('Sorry, but NiceBucket.incCntr() is not implemented yet.')

    def queryCntr(self, cntrIdx=0) -> dict:
        """
        Query a cntr.
        Input:
        cntrIdx - the counter's index.
        Output:
        cntrDic: a dictionary, where:
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.
        """
        print ('Sorry, but ICE_bucket.queryCntr() is not implemented yet.')

