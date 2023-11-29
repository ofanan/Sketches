import math, random, pickle
from printf import printf
import settings
import numpy as np

class CntrMaster (object): 
    """
    Generate, check and parse dynamic "Tetra" counters 
    """

    # Given the size of the "tetra" field, return the size of the "mantissa" field
    mantSizeByTetraSize = lambda self, tetraSize : self.cntrSize - max (tetraSize, 1) - self.calcDelimiterSize (tetraSize=tetraSize) 

    # Given the value of the "tetra" field, return the size of the "mantissa" field
    mantSizeByTetraVal  = lambda self, tetraVal : self.mantSizeByTetraSize (tetraSize=self.tetraValToSize(tetraVal)) 

    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr = lambda self : 'TetraDyn_n{}_t{}' .format (self.cntrSize, self.tetraMaxSize)
    
    # returns the value of a number given its offset, tetra and mant
    valOf = lambda self, offset, mantVal, tetraVal : offset + mantVal*(1 << (1 << tetraVal))

    # The size of the delimiter between the tetra and the mant field is either 0 (if tetraSize==maxTetraSize) or 1 (else)     
    calcDelimiterSize = lambda self, tetraSize : 0 if (self.tetraMaxSize==1 or tetraSize==0) else (0 if tetraSize==self.tetraMaxSize else 1)

    # print the details of the counter in a convenient way 
    printCntrLine = lambda self, cntr, tetraVec, mantVal, cntrVal : print ('cntrVec={}, tetraVec={}, mantVal={} cntrval={}' .format (cntr, tetraVec, mantVal, cntrVal))   

    # Returns the maximum value of the counter with its current params 
    cntrMaxVal          = lambda self : self.cntrMaxVal
     
    # Returns the counter that reaches the max value  
    cntrMaxVec          = lambda self : self.cntrMaxVec

    # Given the values of the mantissa and the tetraSize, returns the binary cntr representing them.
    mantValNTetraSize2cntr = lambda self, mantVal, tetraSize : \
                          '1' * tetraSize + '0' * self.calcDelimiterSize (tetraSize) + \
                          np.binary_repr (num=mantVal, width=self.cntrSize - tetraSize - self.calcDelimiterSize (tetraSize))
    
    mantNTetraVals2cntr = lambda self, mantVal, tetraVal : self.mantValNTetraSize2cntr (mantVal=mantVal, tetraSize=self.tetraValToSize(tetraVal=tetraVal))
    
    def tetraValToSize (self, tetraVal):
        """
        Given the value of tetra field, return its size.
        """
        return self.tetraMaxSize - tetraVal
        
    
    def tetraSizeToVal (self, tetraSize):
        """
        Given the value of tetra field, return its size.
        """
        return self.tetraMaxSize - tetraSize
        
    
    def calcOffsets (self):
        """
        Pre-calculate all the offsets to be added to a counter, according to its tetra value.
        self.offsetOfTetraVal[t] will hold the offset to be added to the counter's val when the tetra value is t.
        """ 
        self.offsetOfTetraVal   = np.zeros (self.tetraMaxVal+1) 
        for tetraVal in range (len(self.offsetOfTetraVal)-1): # in Dyn mode, tetraSize==tetraVal
            mantSize = self.mantSizeByTetraVal (tetraVal=tetraVal)  
            self.offsetOfTetraVal[tetraVal+1] = self.offsetOfTetraVal[tetraVal] + (1 << mantSize) * (1 << (1 << tetraVal))

    def __init__ (self, cntrSize=8, tetraMaxSize=1, numCntrs=1, verbose=[]):
        
        """
        Initialize an array of cntrSize counters. The cntrs are initialized to 0.
        Inputs:
        cntrSize  - num of bits in each counter.
        tetraSize - size of the tetra field, in bits. 
        numCntrs - number of counters in the array.
        verbose - can be either:
            settings.VERBOSE_COUT_CNTRLINE - print to stdout details about the concrete counter and its fields.
            settings.VERBOSE_DEBUG         - perform checks and debug operations during the run. 
            settings.VERBOSE_RES           - print output to a .res file in the directory ../res
            settings.VERBOSE_PCL           = print output to a .pcl file in the directory ../res/pcl_files
            settings.VERBOSE_DETAILS       = print to stdout details about the counter
            settings.VERBOSE_NOTE          = print to stdout notes, e.g. when the target cntr value is above its max or below its min.
        """

        if (cntrSize<3):
            print ('error: cntrSize requested is {}. However, cntrSize should be at least 3.' .format (cntrSize))
            exit ()
        if (tetraMaxSize > (cntrSize-2)):
            print ("You chose tetraMaxSize {} that is too large for cntrSize {}" .format (tetraMaxSize, cntrSize))
            self.isFeasible = False
            return         
        self.isFeasible   = True  # will be False in case of wrong initialization parameters
        self.cntrSize     = int(cntrSize)
        self.numCntrs     = numCntrs
        self.verbose      = verbose
        self.tetraMaxSize = tetraMaxSize
        self.tetraMaxVal =  self.tetraMaxSize #using unary presentation  
        self.cntrZeroVec = '0' * self.cntrSize
        self.cntrMaxVec  = '0' + ('1' * (self.cntrSize-1)) 
        self.rstAllCntrs ()
        self.calcOffsets ()
        self.cntrMaxVal  = self.cntr2num (self.cntrMaxVec)
        
    def rstAllCntrs (self):
        """
        """
        self.cntrs = [self.cntrZeroVec for _ in range (self.numCntrs)]
        
    def rstCntr (self, cntrIdx=0):
        """
        """
        self.cntrs[cntrIdx] = self.cntrZeroVec
        
        
    def cntr2num (self, cntr, verbose=None):
        """
        Convert a counter, given as a binary vector (e.g., "11110"), to an integer num.
        print the counter (if requested by the user's verbose).
        Returns the value of the cntr (as int). 
        """
        
        if (verbose!=None): #if a new verbose was given, it overrides the current verbose
            self.verbose = verbose
        if (len(cntr) != self.cntrSize): # if the cntr's size differs from the default, we have to update the basic params
            settings.error ('the size of the given counter is {} while CntrMaster was initialized with cntrSize={}' .format (len(cntr), self.cntrSize))

        tetraSize     = settings.idxOfLeftmostZero (ar=cntr, maxIdx=self.tetraMaxSize)
        delimiterSize = self.calcDelimiterSize (tetraSize)
        tetraVec      = '1' * tetraSize
        mantVec       = cntr [tetraSize+delimiterSize:]  
        tetraVal      = self.tetraSizeToVal(tetraSize)
        mantVal       = int (mantVec, base=2)
        cntrVal       = self.offsetOfTetraVal[tetraVal] + mantVal * (2**(2**tetraVal))
        if (settings.VERBOSE_COUT_CNTRLINE in self.verbose):
            self.printCntrLine (cntr=cntr, tetraVec=tetraVec, mantVal=mantVal, cntrVal=cntrVal)
        return cntrVal
    
    def queryCntr (self, cntrIdx=0):
        """
        Query a cntr.
        Input: 
        cntrIdx - the counter's index. 
        Output:
        cntrDic: a dictionary, where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.        
        """
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='Tetra dynamic')                
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : self.cntr2num(self.cntrs[cntrIdx])}    
        
    def incCntr (self, cntrIdx=0, factor=1, mult=False, verbose=[]):
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
        - optionalModifiedCntr - an array with entries, representing the counters closest to the target value from below and from above.
          If the target value can be accurately represented by the counter, then optionalModifiedCntr will include 2 identical entries. 
          Each entry in optionalModifiedCntr is a cntrDict that consists of: 
          - cntrDict['cntrVec'] - the binary counter.
          - cntrDict['val']  - the counter's value.
        """
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='TetraDyn')
        targetVal = (self.cntr2num (self.cntrs[cntrIdx]) * factor) if mult else (self.cntr2num (self.cntrs[cntrIdx]) + factor)
        optionalModifiedCntr = self.num2cntr (targetVal)
        if (settings.VERBOSE_DETAILS in verbose): 
            print ('targetVal={}\n, optionalModifiedCntr={}' .format (targetVal, optionalModifiedCntr))
        if (len(optionalModifiedCntr)==1): # there's a single option to modify the cntr -- either because targetVal is accurately represented, or because it's > maxVal, or < 0.
            self.cntrs[cntrIdx] = optionalModifiedCntr[0]['cntrVec']
        else:
            probOfFurtherInc = float (targetVal - optionalModifiedCntr[0]['val']) / float (optionalModifiedCntr[1]['val'] - optionalModifiedCntr[0]['val'])
            self.cntrs[cntrIdx] = optionalModifiedCntr[1]['cntrVec'] if (random.random() < probOfFurtherInc) else optionalModifiedCntr[0]['cntrVec']
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : self.cntr2num(self.cntrs[cntrIdx])}    
            
    def num2cntr (self, targetVal, verbose=None):
        """
        given a target value, find the closest counters to this targetVal from below and from above.
        Output:
        - A list of dictionaries, where, at each entry, 'cntrVec' is the binary counter, 'val' is its integer value.
        - If an exact match was found (the exact targetVal can be represented), the list contains a single dict entry: the cntr representing this targetVal. 
        - If targetVal <= 0, the list has a single dict entry: the cntr representing 0 
        - If targetVal > maxVal that this cntr can represent, the list has a single dict entry: the cntr repesenting maxVal
        - Else, 
            The first entry in the list is the dict of the max cntr value that is < targetVal.
            The second entry is the dict of min cntr val that is > targetVal.
        """
        if (verbose!=None): #if a new verbose was given, it overrides the current verbose
            self.verbose = verbose
        if (targetVal > self.cntrMaxVal):
            if (settings.VERBOSE_NOTE in self.verbose):
                print ('Note: the requested cntr value {} is above the max feasible cntr for this configuration' .format(targetVal))
            return [{'cntrVec' : self.cntrMaxVec, 'val' : self.cntrMaxVal}]
        if (targetVal < 0):
            if (settings.VERBOSE_NOTE in self.verbose):
                print ('Note: the requested cntr value {} is negative' .format (targetVal))
            return [{'cntrVec' : self.cntrZeroVec, 'val' : 0}]
        
        offset   = max ([offset for offset in self.offsetOfTetraVal if offset<=targetVal])
        tetraVal = list(self.offsetOfTetraVal).index(offset)
        mantSize = self.mantSizeByTetraVal (tetraVal) 
        mantVal  = math.floor (float(targetVal-offset)/float(1 << (1 << tetraVal)))
        cntr     = self.mantNTetraVals2cntr (mantVal, tetraVal)       
        cntrVal  = self.valOf(offset, mantVal, tetraVal)
        if (cntrVal==targetVal): # found a cntr that accurately represents the target value
            return [{'cntrVec' : cntr, 'val' : cntrVal}]

        # now we know that the counter found is < the target value
        if (mantVal < (1 << mantSize) -1): 
            cntrpp    = self.mantNTetraVals2cntr (mantVal+1, tetraVal)
            cntrppVal = self.valOf(offset, mantVal+1, tetraVal)
        else: 
            cntrpp    = self.mantNTetraVals2cntr (mantVal=0, tetraVal=tetraVal+1)
            cntrppVal = self.valOf(offset=self.offsetOfTetraVal[tetraVal+1], mantVal=0, tetraVal=tetraVal+1)
        return [{'cntrVec' : cntr, 'val' : cntrVal}, {'cntrVec' : cntrpp, 'val' : cntrppVal}]        
        
def printAllVals (cntrSize=8, tetraMaxSize=2, verbose=[settings.VERBOSE_RES]):
    """
    Loop over all the binary combinations of the given counter size. 
    For each combination, print to file the respective counter, and its value. 
    The prints are sorted in an increasing order of values.
    """
    print ('running printAllVals. mode=Tetra dynamic')
    myCntrMaster = CntrMaster(cntrSize=cntrSize, tetraMaxSize=tetraMaxSize)
    listOfVals = []
    for i in range (2**cntrSize):
        cntr = np.binary_repr(i, cntrSize) 
        val = myCntrMaster.cntr2num(cntr)
        listOfVals.append ({'cntrVec' : cntr, 'val' : val})
    listOfVals = sorted (listOfVals, key=lambda item : item['val'])

    if (settings.VERBOSE_RES in verbose):
        outputFile    = open ('../res/{}.res' .format (myCntrMaster.genSettingsStr()), 'w')
        for item in listOfVals:
            printf (outputFile, '{}={:.0f}\n' .format (item['cntrVec'], item['val']))
      

def printAllCntrMaxVals (cntrSizeRange=[], tetraMaxSizeRange=None, verbose=[settings.VERBOSE_RES]):
    """
    print the maximum value a cntr reach for several "configurations" -- namely, all combinations of cntrSize and hyperSize. 
    """

    if (settings.VERBOSE_RES in verbose):
        outputFile    = open ('../res/cntrMaxVals.txt', 'a')
    for cntrSize in cntrSizeRange:
        for tetraMaxSize in range (3) if tetraMaxSizeRange==None else tetraMaxSizeRange:
            myCntrMaster = CntrMaster(cntrSize=cntrSize, tetraMaxSize=tetraMaxSize)
            if (not (myCntrMaster.isFeasible)):
                continue
            if (myCntrMaster.cntrMaxVal < 10**8):
                printf (outputFile, '{} cntrMaxVal={:.0f}\n' .format (myCntrMaster.genSettingsStr(), myCntrMaster.cntrMaxVal))
            else:
                printf (outputFile, '{} cntrMaxVal={}\n' .format (myCntrMaster.genSettingsStr(), myCntrMaster.cntrMaxVal))


# myCntrMaster = CntrMaster(cntrSize=5)
# printAllCntrMaxVals (cntrSizeRange=[13, 14, 15, 16], tetraMaxSizeRange=range (1,5), verbose=[settings.VERBOSE_RES])
# printAllVals (cntrSize=8, tetraMaxSize=1, verbose=[settings.VERBOSE_RES])
# printAllVals (cntrSize=8, tetraMaxSize=3, verbose=[settings.VERBOSE_RES])
