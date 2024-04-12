# This class implements a single ICE_bucket.
# ICE_buckets are detailed in the paper: "Independent counter estimation buckets", Einziger, Gil and Fellman, Benny and Kassner, Yaron, Infocom'12.
import random, math, numpy as np
from printf import printf
import settings

# To prevent overflows, the search range for some parameters should be limited corresponding to the counter's size. 
# This is done using the list of dicts below.
# The data below is used also to determine the 'epsilonStep' used in ICE_buckets. 
preComputedData = [
                 {'cntrSize' : 4,    'epsilonStep' : 0.15},
                 {'cntrSize' : 5,    'epsilonStep' : 0.09},
                 {'cntrSize' : 6,    'epsilonStep' : 0.055},
                 {'cntrSize' : 7,    'epsilonStep' : 0.035},
                 {'cntrSize' : 8,    'epsilonStep' : 0.0302},
                 {'cntrSize' : 9,    'epsilonStep' : 0.016},
                 {'cntrSize' : 10,   'epsilonStep' : 0.014},
                 {'cntrSize' : 11,   'epsilonStep' : 0.012},
                 {'cntrSize' : 12,   'epsilonStep' : 0.0065},
                 {'cntrSize' : 13,   'epsilonStep' : 0.0035},
                 {'cntrSize' : 14,   'epsilonStep' : 0.0024},
                 {'cntrSize' : 15,   'epsilonStep' : 0.0016},
                 {'cntrSize' : 16,   'epsilonStep' : 0.0011},
                 ]

def findPreComputedDatum (cntrSize):
    """
    Returns the precomputed datum with the requested cntrSize.
    """
    preComputedDatum = [item for item in preComputedData if item['cntrSize']==cntrSize]
    if len(preComputedDatum)==0:
        settings.error ('Sorry, but the requested cntrSize {cntrSize} is currently not supported by CEDAR')
    elif len(preComputedDatum)>1:
        settings.error ('More then one entry in preComputedData for the requested cntrSize {cntrSize}')
    return preComputedDatum[0]
           
def calcEstimatorGivenEpsilon (epsilon, ell):
    """
    calculate the ell-th estimator, given epsilon, using (1) from [ICE_buckets]. 
    The corresponding .tex code is:
    \frac{\left(\left(1+2\cdot \:\:x^2\right)^\ell-1\right)}{2x^2}\left(1+x^2\right) 
    """
    if epsilon<0 or ell<0: 
        settings.error (f'in IceBucket:calcEstimatorGivenEpsilon(). epsilon={epsilon}, ell={ell}')
    elif epsilon==0: # perfect estimator - identity function
        return ell
    return int ((((1+2*epsilon**2)**ell -1)/(2*epsilon**2)) * (1 + epsilon**2))
    
def calcCntrMaxValsByCntrSizes (numEpsilonSteps=6, cntrSize=4):
    """
    Given the counter's size, find the pre-computed epsilonStep.
    For each value of epsilon in [0, epsilonStep, 2*epsilonStep, 3*epsilonStep, ...], 
    calculate the max counter's val.
    Return an array with the max counter's val.
    """
    epsilonStep = findPreComputedDatum (cntrSize)['epsilonStep']
    # print (f'cntrSize={cntrSize}')
    res = [None] * numEpsilonSteps
    epsilon = 0 
    for step in range (numEpsilonSteps):
        res[step] = calcCntrMaxValGivenEpsilon(epsilon, cntrSize)
        epsilon  += epsilonStep 
    print (f'maxVals={res}')
    return res

calcCntrMaxValGivenEpsilon = lambda epsilon, cntrSize : calcEstimatorGivenEpsilon (epsilon=epsilon, ell=(1 << cntrSize) - 1) 

class CntrMaster(object):
    """
    Generate, check and parse counters
    """
    # Generates a string that details the counter's settings (param vals).
    genSettingsStr = lambda self : 'Ice_n{}'.format(self.cntrSize)
    
    def __init__(self, 
                 cntrSize           = 8, # num of bits in each counter.
                 numCntrs           = 1, # number of counters in the array.
                 numEpsilonSteps    = None,    # number of different possible estimation scales - a power of two.
                 cntrMaxVal         = None, # Max value to be reached by a counter. 
                 verbose            = [],
                 id                 = None,
                 ):
        """
        Initialize an array of cntrSize counters. The cntrs are initialized to 0.
        """
        self.cntrSize, self.numCntrs, self.cntrMaxVal = cntrSize, numCntrs, cntrMaxVal
        self.id, self.verbose = id, verbose
        self.numEstimators = 2**self.cntrSize
        self.mode = 'ICE'
        self.rst () # reset all the counters
        self.numEpsilonSteps  = numEpsilonSteps
 
        if self.cntrMaxVal==None:       
            self.epsilon    = 0
            self.epsilonStep      = findPreComputedDatum (cntrSize=self.cntrSize)['epsilonStep']
        else:
            self.calcEpsilonM() 
            self.epsilonStep = self.epsilonM / (self.numEpsilonSteps-1) # Proof of Theorem 4 in [ICE_buckets].
            self.epsilon     = self.epsilonStep 
    
    def calcAllEstimatorsByEpsilon (self):
        """
        Calculate the estimators' values based on the epsilonStep accuracy parameter, as detailed in the paper ICE_buckets.
        """
        if self.epsilon<0: 
            settings.error (f'in IceBucket:calcAllEstimatorsByEpsilon(). epsilon={self.epsilon}')
        elif self.epsilon==0: # perfect estimator - identity function
            return [int(ell) for ell in range (self.numEstimators)]
        else:
            return [int ((((1+2*self.epsilon**2)**ell -1)/(2*self.epsilon**2)) * (1 + self.epsilon**2)) for ell in range (self.numEstimators)] 
        

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
            settings.error ('in IceBucket.rst() : sorry, cntrSize>32 is not supported yet.')

    def calcEpsilonM (self):
        """
        Given the requested max counter val (M), calculate Epsilon resulting in this max counter val.
        The calculation is done using binary search.
        The resulted value is assigned to self.epsilonM (see Sec. III.C. in [ICE_buckets]
        """
        if self.cntrSize <= 8: 
            epsilonLo, epsilonHi, binSearhResolution = 0.01, 1, 0.001
        elif self.cntrSize <= 12: 
            epsilonLo, epsilonHi, binSearhResolution = 0.0001, 0.1, 0.0001
        elif self.cntrSize <= 14: 
            epsilonLo, epsilonHi, binSearhResolution = 0.00001, 0.01, 0.00001
        elif self.cntrSize <= 16: 
            epsilonLo, epsilonHi, binSearhResolution = 0.000001, 0.001, 0.000001
        else:
            settings.error ('in CEDAR.calcEpsilonM() : sorry, cntrSize>16 is not supported yet in ICE buckets.')
        
        if calcCntrMaxValGivenEpsilon (epsilonHi, self.cntrSize) < self.cntrMaxVal:
            setting.serror (f'in CEDAR.calcEpsilonM. Could not reach cntrMaxVal={self.cntrMaxVal} with cntrSize={self.cntrSize} even with the highest suggested Epsilon={epsilonHi}.')
            return
        
        self.epsilonM        = epsilonHi
        
        while (True):
            if (epsilonHi - epsilonLo < binSearhResolution): # converged. Still, need to check whether this epsilon is high enough.
                if calcCntrMaxValGivenEpsilon (self.epsilonM, self.cntrSize) >= self.cntrMaxVal: # can reach maxVal with this epsilon --> Good
                    return
                # now we know that cannot reach targetMaxVal with the current epsilon
                self.epsilonM += binSearhResolution
                if calcCntrMaxValGivenEpsilon (self.epsilonM, self.cntrSize) < self.cntrMaxVal: 
                    settings.error ('in CEDAR.calcEpsilonM. problem at binary search')
                return
                
            self.epsilonM = (epsilonLo + epsilonHi)/2
            maxValOfThisEpsilon = calcCntrMaxValGivenEpsilon (self.epsilonM, self.cntrSize)
            if (settings.VERBOSE_DETAILS in self.verbose):
                printf (self.detailFile, 'epsilon={}\n' .format (self.epsilonM, self.cntrSize))
            
            if maxValOfThisEpsilon==self.cntrMaxVal: # found exact match 
                break
            if maxValOfThisEpsilon < self.cntrMaxVal: # can't reach maxVal with this epsilon --> need larger epsilon value
                epsilonLo = self.epsilonM
            else: # maxVal > targetMaxVal --> reached the maximum value - try to decrease epsilon, to find a tighter value.
                epsilonHi = self.epsilonM

        
    def rstCntr (self, cntrIdx=0):
        """
        """
        self.cntrs[cntrIdx] = 0


    def upScale (self):
        """
        Up-scale for reaching a largest maximal value. In particular:
        - Increase the self.epsilon, which determines the error, by self.epsilonStep. Increasing self.epsilon allows reaching larger counted value (at the cost of a larger relative error).
        - calculate the estimators' values using the updated self.epsilon. (localUpscale procedure, defined in [ICE_buckets]).   
        - For each counter ("symbol"), run the "symbol upsclae" procedure, defined in [ICE_buckets].
          This procedure scales-up a single counter after the "epsilon" variable was increased.
        """        
        if self.epsilon == ( (self.numEpsilonSteps-1) * self.epsilonStep):
            settings.error (f'IceBucket.upScale() called when epsilon is already maximal. Cannot further increase epsilon. numEpsilonSteps={self.numEpsilonSteps}. Max val is {calcCntrMaxValsByCntrSizes(cntrSize=self.cntrSize)[-1]}')

        # Update self.epsilon and then update all the estimators' values accordingly.
        self.prevEpsilon    = self.epsilon  
        self.epsilon       += self.epsilonStep
        self.calcAllEstimatorsByEpsilon () 

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
        Increase a counter cntrIdx by a 1 and return the updated value.
        """
        cntrVal = self.cntrs[cntrIdx]
        if cntrVal==(1 << self.cntrSize) - 1: # reached the largest possible estimated value w/o up-scaling?
            if settings.VERBOSE_LOG in self.verbose:
                printf (self.logFile, f'bkt {self.id} is up-scaling. epsilon b4 upscaling={self.epsilon}\n')
            self.upScale () 
            cntrVal = self.cntrs[cntrIdx]# cntrVal is the value in the counter after up-scaling, before incrementing
        curEstimate = calcEstimatorGivenEpsilon(self.epsilon, ell=cntrVal)
        incEstimate = calcEstimatorGivenEpsilon(self.epsilon, ell=cntrVal+1)
        diff = incEstimate - curEstimate  
        if diff==1 or random.random () < 1/diff: 
            self.cntrs[cntrIdx] += 1
            return incEstimate
        return curEstimate

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
        print ('Sorry, but ICE_bucket.incCntr() is not implemented yet.')

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

    def getAllCntrsVals (self) -> list:
        """
        Returns a vector containing all the counters' values
        """
        vals = [None]*self.numCntrs
        for cntrNum in range(self.numCntrs):
            vals[cntrNum] = calcEstimatorGivenEpsilon(self.epsilon, self.cntrs[cntrNum])
        return vals
    
    def printAllCntrVals (self, outputFile=None) -> None:
        """
        Format-print all the counters as a single the array, to the given file.
        """
        if outputFile==None:
            print ('cntrs={} ' .format([calcEstimatorGivenEpsilon (self.epsilon, cntr) for cntr in self.cntrs]))
        else:
            printf (outputFile, f'bkt id={self.id}, Estep={self.epsilon/self.epsilonStep}\n[')
            for cntr in self.cntrs:
                printf (outputFile, '{:.0f} ' .format(calcEstimatorGivenEpsilon(self.epsilon, cntr)))
            printf (outputFile, f']\n')

    def printEstimators (self, outputFile=None) -> None:
        """
        Generate and format-print all the counters as a single the array, to the given file.
        """
        estimators = self.calcAllEstimatorsByEpsilon()
        if outputFile==None:
            print ('eps={:.3f}, estimators={}' .format (self.epsilon, estimators))            
        else:
            for cntr in self.cntrs:
                printf (outputFile, '{:.0f} ' .format(calcCntrMaxValGivenEpsilon(self.epsilon, self.cntrSize)))
    
    
# calcCntrMaxValsByCntrSizes (numEpsilonSteps=6, cntrSize=8)
