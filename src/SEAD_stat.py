import math, time, random
from printf import printf
import settings
import numpy as np

class CntrMaster (object):
    """
    Generate, check and parse counters
    """

    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr = lambda self : 'SEADstat_n{}_e{}' .format (self.cntrSize, self.expSize) # if self.mode=='stat' else 0)
    
    # print the details of the counter in a convenient way
    printCntrLine       = lambda self, cntr, expVec, expVal, mantVec, mantVal, cntrVal : print ('expVec={}, expVal={}, mantVec={}, mantVal={}, offset={}, val={}'
                                                                                               .format (expVec, expVal, mantVec, mantVal, self.offsetOfExpVal[expVal], cntrVal))    
    # returns the value of a cntr given its exp and mant
    valOf = lambda self, mantVal, expVal : self.offsetOfExpVal[expVal] + mantVal*2**expVal
     
    # increment a binary vector, regardless the partition to mantissa, exponent etc.
    # E.g., given a binary vec "00111", this func' will return "01000"  
    incBinVec = lambda self, vec, delta=1 : np.binary_repr (int(vec, base=2)+delta, len(vec)) 

    # get the mantisa vector  
    getMantVec = lambda self, cntrIdx : self.cntrs[cntrIdx][self.expSize:]
                
    # get the exponent vector  
    getExpVec  = lambda self, cntrIdx : self.cntrs[cntrIdx][:self.expSize]            

    # get the exponent value   
    getExpVal  = lambda self, cntrIdx : int (self.getExpVec(cntrIdx), base=2)             

    # get the mantissa value in 'stat' mode  
    getMantVal = lambda self, cntrIdx : int (self.cntrs[cntrIdx][self.expSize:], base=2)
    
    def calcOffsets (self):
        """
        Pre-calculate all the offsets to be added to a counter, according to its exponent value:
        self.offsetOfExpVal[e] will hold the offset to be added to the counter's val when the exponent's value is e.
        """
        self.offsetOfExpVal   = np.zeros (self.expMaxVal+1)  
        for expVal in range (self.expMaxVal): # for each potential exponent value
            self.offsetOfExpVal[expVal+1] = self.offsetOfExpVal[expVal] + 2**(expVal+self.mantSize)
  
    def __init__ (self, 
                  cntrSize=4,   # num of bits in each counter.
                  expSize=2,    # size of the exp field, in bits. Relevant only for static counters.
                  numCntrs=1,   # number of counters in the array. 
                  verbose=[]    # one of the verbose macros, detailed in settings.py
                  ):
        """
        Initialize an array of cntrSize counters. The cntrs are initialized to 0.
        """
        
        if (cntrSize<3):
            settings.error ('error: cntrSize requested is {}. However, cntrSize should be at least 3.' .format (cntrSize))
        self.cntrSize    = int(cntrSize)
        self.numCntrs    = int(numCntrs)
        self.verbose     = verbose
        self.cntrZeroVec = '0' * self.cntrSize
        self.cntrs       = [self.cntrZeroVec] * self.numCntrs
        self.expSize     = expSize
        self.calcParams ()
        if settings.VERBOSE_LOG_CNTRLINE in self.verbose:
            self.logFIle = open (f'../res/log_files/{self.genSettingsStr()}.log', 'w')

             
    def rstCntr (self, cntrIdx=0):
        """
        """
        self.cntrs[cntrIdx] = self.cntrZeroVec
        
    def calcParams (self):
        """
        Pre-compute the cntrs' parameters, in case of a static SEAD cntr 
        """
        if (self.expSize >= self.cntrSize):
            print ('error: for cntrSize={}, the maximal allowed expSize is {}' .format (self.cntrSize, self.expSize))
            exit  ()
        self.cntrMaxVec = '1' * self.cntrSize
        self.mantSize   = self.cntrSize - self.expSize
        self.expMaxVal  = 2**self.expSize - 1
        self.calcOffsets ()
        self.cntrMaxVal = self.valOf (mantVal=2**self.mantSize-1, expVal=self.expMaxVal)
   
    def cntr2num (self, 
                  cntr, # the counter, given as a binary vector (e.g., "11110"). 
                  ):
        """
        Convert a counter, given as a binary vector (e.g., "11110"), to an integer num.
        Output: integer.
        """        
        if (len(cntr) != self.cntrSize): # if the cntr's size differs from the default, we have to update the basic params
            print ('the size of the given counter is {} while CntrMaster was initialized with cntrSize={}.' .format (len(cntr), self.cntrSize))
            print ('Please initialize a cntr with the correct len.')
            exit ()        
        expVec  = cntr[:self.expSize]
        mantVec = cntr[self.expSize:]
        if (settings.VERBOSE_COUT_CNTRLINE in self.verbose):
            expVal  = int (expVec, base=2)
            mantVal = int (mantVec, base=2)
            cntrVal = self.valOf (expVal=expVal, mantVal=mantVal)
            self.printCntrLine (cntr=cntr, expVec=expVec, expVal=expVal, mantVec=mantVec, mantVal=mantVal, cntrVal=cntrVal)
        return self.valOf (expVal=int (expVec, base=2), mantVal=int (mantVec, base=2))

    
    def queryCntr (self, cntrIdx=0) -> dict:
        """
        Query a cntr.
        Input: 
        cntrIdx - the counter's index. 
        Output:
        cntrDic: a dictionary, where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.        
        """
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='SEAD')
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : self.cntr2num(self.cntrs[cntrIdx])}    
        
    def incCntr (self, cntrIdx=0, factor=int(1), mult=False, verbose=None):
        """
        """
        if verbose!=None:
            self.verbose = verbose
        if factor==1 and mult==False:
            return self.incCntrBy1GetVal (cntrIdx)
    
        settings.error ('Sorry. SEAD_stat.inccntr() is currently implemented only as incCntrBy1.')
    
    def incCntrBy1GetVal (self, cntrIdx=0):
        """
        Increase a counter by 1.
        Operation:
        Define cntrVal as the current counter's value. 
        Then, targetValue = cntrVal+1  
        If targetValue > maximum cntr's value, return the a cntr representing the max possible value. 
        If targetValue < 0, return a cntr representing 0.
        If targetValue can be represented correctly by the counter, return the exact representation.
        Else, use probabilistic cntr's modification.
        Return the updated cntr's value.
        """
        cntrCurVal = self.cntr2num (self.cntrs[cntrIdx])
        if cntrCurVal == self.cntrMaxVal:
            return cntrCurVal
            # return {'cntrVec' : self.cntrs[cntrIdx], 'val' : cntrCurVal}    

        expVal  = self.getExpVal (cntrIdx)
        cntrppVal = cntrCurVal + 2**expVal

        if random.random() >= 1/float(cntrppVal-cntrCurVal):
            return cntrCurVal 
            # return {'cntrVec' : self.cntrs[cntrIdx], 'val' : cntrCurVal}    

        # Need to increment the cntr
        mantVal = self.getMantVal(cntrIdx)
        if (mantVal < 2**self.mantSize-1): # can we further increment the mantissa w/o o/f?
            self.cntrs[cntrIdx] = np.binary_repr(expVal, self.expSize) + np.binary_repr (mantVal+1, self.mantSize)
        else:  # need to increase the exponent
            self.cntrs[cntrIdx] = np.binary_repr(expVal+1, self.expSize) + '0' * self.mantSize # need to decrement the mantissa field size.
        if settings.VERBOSE_LOG_CNTRLINE in self.verbose:
            printf (self.logFIle, f'After inc: cntrVec={self.cntrs[cntrIdx]}, cntrVal={cntrppVal}\n')
        return cntrppVal
        # return {'cntrVec' : self.cntrs[cntrIdx], 'val' : cntrppVal}    


def printAllVals (cntrSize=4, expSize=1, verbose=[]):
    """
    Loop over all the binary combinations of the given counter size. 
    For each combination, print to file the respective counter, and its value. 
    The prints are sorted in an increasing order of values.
    """
    print ('running SEAD.printAllVals()')
    listOfVals = []
    myCntrMaster = CntrMaster(cntrSize=cntrSize, expSize=expSize, verbose=verbose)
    for i in range (2**cntrSize):
        cntr = np.binary_repr(i, cntrSize) 
        listOfVals.append ({'cntrVec' : cntr, 'val' : myCntrMaster.cntr2num(cntr)})
    listOfVals = sorted (listOfVals, key=lambda item : item['val'])

    if (settings.VERBOSE_RES in verbose):
        myCntrMaster.cntrSize   = cntrSize
        myCntrMaster.expSize    = expSize
        outputFile    = open ('../res/log_files/{}.res' .format (myCntrMaster.genSettingsStr()), 'w')
        for item in listOfVals:
            printf (outputFile, '{}={}\n' .format (item['cntrVec'], item['val']))
    print ('cntrMaxVal={}' .format (myCntrMaster.cntrMaxVal))

def printAllCntrMaxVals (cntrSizes=[], expSizes=None, verbose=[settings.VERBOSE_RES]):
    """
    print the maximum value a cntr reach for several "configurations" -- namely, all combinations of cntrSize and hyperSize. 
    """

    if (settings.VERBOSE_RES in verbose):
        outputFile    = open ('../res/cntrMaxVals.txt', 'a')
    for cntrSize in cntrSizes:
        expSizes = expSizes if (expSizes!=None) else range (1, cntrSize)
        for expSize in expSizes:
            # print ('cntrSize={}, expSize={}' .format (cntrSize, expSize))
            myCntrMaster = CntrMaster (cntrSize=cntrSize, expSize=expSize)
            printf (outputFile, '{} cntrMaxVal={:.0f}\n' .format (myCntrMaster.genSettingsStr (cntrSize=cntrSize, expSize=expSize), myCntrMaster.cntrMaxVal))

def checkTimes ():
    """
    check which code style is faster.
    The tests show that shift is slightly slower than mult.
    """
    cntrSize = 16
    mantVal = 1
    
    startTime = time.time ()
    for _ in range (50):
        for i in range (2**cntrSize):
            cntr = np.binary_repr(i, cntrSize) 
            for expSize in range (1, 4):
                expVal  = int (cntr[:expSize], base=2)
                cntrVal = mantVal*2**expVal
    print ('t by mult={}' .format (time.time()-startTime))

    startTime = time.time ()
    for _ in range (50):
        for i in range (2**cntrSize):
            cntr = np.binary_repr(i, cntrSize) 
            for expSize in range (1, 4):
                expVal  = int (cntr[:expSize], base=2)
                cntrValByShift = mantVal << expVal
    print ('t by shift={}' .format (time.time()-startTime))

