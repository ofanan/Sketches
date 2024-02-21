# This file implements an FP with parametric size of mantissa and exponent fields.  
import math, random, pickle
from printf import printf
import settings
import numpy as np

class CntrMaster (object):
    """
    Generate, check and perform arithmetic operations on F2P counters in SR (Small Reals) flavors.
    The counters are generated as an array with the same format (counterSize and hyperExpSize).
    Then, it's possible to perform arithmetic ops on the counters in chosen indices of the array. 
    """

    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr = lambda self : f'FP_n{self.cntrSize}_m{self.mantSize}_e{self.expSize}'
    
    # print the details of the counter in a convenient way
    printCntrLine  = lambda self, cntr, expVec, expVal, power, mantVec, mantVal, cntrVal : print (f'cntr={cntr}, hyperVec={cntr[0:self.hyperSize]}, expVec={expVec}, bias={self.bias}, expVal={expVal}, power={power}, mantVec={cntrVal}, mantVal={mantVal}, val={cntrVal}')

    def __init__ (self, 
                  cntrSize  = 8, # of bits in the cntr 
                  expSize   = 2,
                  signed    = False, # When True, the FP is signed (the MSB is the sign bit)
                  numCntrs  = 1, # of cntrs in the cntrs' array
                  verbose   = []    # the optional verbose values are detailed in settings.py
                  ):
        
        """
        Initialize an array of cntrSize counters. The cntrs are initialized to 0.
        If the parameters are invalid (e.g., infeasible cntrSize), return None. 
        """
        self.isFeasible = True
        if (cntrSize<3):
            print ('error: cntrSize requested is {}. However, cntrSize should be at least 3.' .format (cntrSize))
            self.isFeasible = False
            return 
        self.cntrSize   = int(cntrSize)
        self.signed     = signed
        self.numCntrs   = numCntrs
        self.verbose    = verbose
        self.expSize    = expSize
        if self.expSize + 1 > self.cntrSize: # need at least 1 mantissa bit
            self.isFeasible = False
            return 
        self.cntrZeroVec    = '0'*self.cntrSize
        self.cntrMaxVec     = '1'*self.cntrSize
        self.cntrMaxVal     = self.cntr2num (self.cntrMaxVec)
        self.bias           = 2**(self.expSize-1)
        settings.error (self.bias) #$$$
        if settings.VERBOSE_COUT_CONF in self.verbose:
            print (self.genSettingsStr ())
        self.rstAllCntrs ()
        
    def rstAllCntrs (self):
        """
        """
        self.cntrs = [self.cntrZeroVec for _ in range (self.numCntrs)]
        if self.signed: # use signed counters - add an array of the signs
            self.signs = np.full(self.numCntrs, True)
        
    def rstCntr (self, cntrIdx=0):
        """
        """
        self.cntrs[cntrIdx] = self.cntrZeroVec
        if self.signed: 
            self.signs[cntrIdx] = True
        
    def cntr2num (self, cntr):
        """
        Given a counter, as a binary vector (e.g., "11110"), return the number it represents.
        """
        if (len(cntr) != self.cntrSize): # if the cntr's size differs from the default, we have to update the basic params
            settings.error (f'In FP.cntr2num(). the size of the given counter {cntr} is {len(cntr)} while CntrMaster was initialized with cntrSize={self.cntrSize}')

        expVec  = cntr[0:self.expSize]
        mantVec = cntr[self.expSize:]
        mantVal = float (int (mantVec, base=2)) / 2**(self.cntrSize - self.expSize)  
        if expVec == '0'*self.expSize:
            cntrVal  = mantVal * (2**(self.bias+1))
        else:
            cntrVal  = (1 + mantVal) * (2**(int(expVec, self.expSize)+self.bias))
        if settings.VERBOSE_COUT_CNTRLINE in self.verbose:
            expVal = int(expVec, self.expSize)
            if expVec == '0'*self.expSize:
                power = self.bias+1
            else:
                power = expVal + self.bias
            self.printCntrLine (cntr=cntr, expVec=expVec, expVal=expVal, power=power, mantVec=mantVec, mantVal=mantVal, cntrVal=cntrVal)
        if self.signed and self.signs[cntrIdx]==False:
            cntrVal *= (-1)
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
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='F2P_new')        
        
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : self.cntr2num(self.cntrs[cntrIdx])}    
        

def printAllVals(cntrSize=8, expSizes=None, verbose=[]):
    """
    Loop over all the binary combinations of the given counter size.
    For each combination, print to file the respective counter, and its value.
    The prints are sorted in an increasing order of values.
    """
    listOfVals = []
    expSizes = range (1, cntrSize) if expSizes==None else expSizes
    for expSize in expSizes: 
        myCntrMaster = CntrMaster(cntrSize=cntrSize, expSize=expSize)
        val = myCntrMaster.cntr2num(num)
        for num in range(2 ** cntrSize):
            listOfVals.append ({'cntrVec' : np.binary_repr(num, cntrSize), 'val' : val})
    
    if settings.VERBOSE_RES in verbose:
        outputFile = open('../res/{}.res'.format(myCntrMaster.genSettingsStr()), 'w')
        for item in listOfVals:
            printf(outputFile, '{}={:.1f}\n'.format(item['cntrVec'], item['val']))


printAllVals (cntrSize=8, expSizes=None, verbose=[])
