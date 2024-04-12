# This file implements F2P LR, namely, F2P flavor that focuses on improved accuracy on large reals. 
# For futher details, see "main.tex" in Cntr's Overleaf project.
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

    flavor = lambda self : 'sr'
    
    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr = lambda self : f'F2P{self.flavor()}_n{self.cntrSize}_h{self.hyperSize}'
    
    # print the details of the counter in a convenient way
    printCntrLine  = lambda self, cntr, expVec, expVal, power, mantVec, mantVal, cntrVal : print (f'cntr={cntr}, hyperVec={cntr[0:self.hyperSize]}, expVec={expVec}, bias={self.bias}, expVal={expVal}, power={power}, mantVec={cntrVal}, mantVal={mantVal}, val={cntrVal}')

    # Given the vector of the exponent, calculate the value it represents 
    expVec2expVal  = lambda self, expVec, expSize : (2**expSize - 1 + int (expVec, base=2)) if expSize>0 else 0    
    
    # Given the value of the exponent, return the exponent vector representing this value 
    # expVal2expVec       = lambda self, expVal, expSize : np.binary_repr(num=int(self.biasOfExpSize[int(expSize)]) - expVal, width=expSize) if expSize>0 else ""   

    def setFlavorParams (self):
        """
        set variables that are unique for 'sr' flavor of F2P.
        """
        self.bias       = -0.5*(self.Vmax+1)
        self.expMinVec  = ''
        self.expMinVal  = 0
        self.cntrZeroVec = np.binary_repr(0, self.cntrSize)  
        self.cntrMaxVec  = np.binary_repr((1<<self.cntrSize)-1, self.cntrSize)  
            
    def calcParams (self):
        """
        Calc the basics param, which are depended upon the counter size, and the hyper-exp' size.
        """
        self.mantMinSize = self.cntrSize - self.hyperSize - self.expMaxSize 
        if (self.mantMinSize<1):
            print (f'cntrSize={self.cntrSize} and hyperSize={self.hyperSize} implies min mantissa size={self.mantMinSize}. Mantissa size should be at least 1. Please use a smaller hyperSize')
            return False
        self.Vmax = 2**(self.expMaxSize+1)-1 # sum ([2**i for i in range (1, self.expMaxSize+1)])
        self.setFlavorParams ()
        self.powerMin = self.expMinVal + self.bias + 1 
        return True
   
    def __init__ (self, 
                  cntrSize  : int = 8, # of bits in the cntr 
                  hyperSize : int = 1, # of bits in the hyper-exp field 
                  numCntrs  : int = 1, # of cntrs in the cntrs' array
                  verbose   = []    # the optional verbose values are detailed in settings.py
                  ):
        
        """
        Initialize an array of cntrSize counters. The cntrs are initialized to 0.
        If the parameters are invalid (e.g., infeasible cntrSize), return None. 
        """
        self.isFeasible = True
        self.cntrSize   = cntrSize
        if (self.cntrSize<3):
            print ('error: cntrSize requested is {}. However, cntrSize should be at least 3.' .format (cntrSize))
            self.isFeasible = False
            return 
        self.numCntrs   = numCntrs
        self.verbose    = verbose
        if (not (self.setHyperSize (hyperSize))):
            self.isFeasible = False
            return 
        if (not self.calcParams()): # parameters couldn't be calculated, e.g. due to wrong given combination of cntrSize and hyperSize
            self.isFeasible = False
            return 
        # self.cntrMinVal = self.cntr2num (self.cntrZeroVec)
        self.cntrMaxVal = self.cntr2num (self.cntrMaxVec)
        if settings.VERBOSE_COUT_CONF in self.verbose:
            print (f'F2P{self.flavor()}, cntrSize={self.cntrSize}, hyperSize={self.hyperSize}, Vmax={self.Vmax}, bias={self.bias}, zeroVec={self.cntrZeroVec}, maxVec={self.cntrMaxVec}, maxVal={self.cntrMaxVal}, expMinVec={self.expMinVec}, expMinVal={self.expMinVal}')
        self.rstAllCntrs ()
        
    def rstAllCntrs (self):
        """
        """
        self.cntrs = [self.cntrZeroVec for _ in range (self.numCntrs)]
        
    def rstCntr (self, cntrIdx=0):
        """
        """
        self.cntrs[cntrIdx] = self.cntrZeroVec
        
        
    def cntr2num (self, cntr):
        """
        Given a counter, as a binary vector (e.g., "11110"), return the number it represents.
        """
        if (len(cntr) != self.cntrSize): # if the cntr's size differs from the default, we have to update the basic params
            settings.error (f'In F2P_{self.flavor()}.cntr2num(). the size of the given counter {cntr} is {len(cntr)} while CntrMaster was initialized with cntrSize={self.cntrSize}')

        hyperVec = cntr [0:self.hyperSize] 
        expSize = int(hyperVec, base=2) #(cntr [0:self.hyperSize],base=2) 
        expVec  = cntr[self.hyperSize:self.hyperSize+expSize]
        mantVec = cntr[self.hyperSize+expSize:]
        mantVal = float (int (mantVec, base=2)) / 2**(self.cntrSize - self.hyperSize - expSize)  
        if expVec == self.expMinVec:
            cntrVal  = mantVal * (2**self.powerMin)
        else:
            cntrVal  = (1 + mantVal) * (2**(self.expVec2expVal(expVec, expSize)+self.bias))
        if settings.VERBOSE_COUT_CNTRLINE in self.verbose:
            expVal = self.expVec2expVal(expVec, expSize)
            if expVal == self.expMinVal:
                power = self.powerMin
            else:
                power = expVal + self.bias
            self.printCntrLine (cntr=cntr, expVec=expVec, expVal=expVal, power=power, mantVec=mantVec, mantVal=mantVal, cntrVal=cntrVal)
        return cntrVal
    
    def setHyperSize (self, hyperSize):
        """ 
        Sets the size of the hyper-exponent field in F2P counters as follows.
        - Check whether the hyper-exponent field size is feasible.
        - If yes - assign the relevant "self" fields (exponent's field max-size). return True
        - If not - print an error msg and return False
        """
        if (hyperSize<1 or hyperSize>self.cntrSize-2):
            print ('Requested hyperSize {} is not feasible for counter size {}' .format (hyperSize, self.cntrSize))
            return False
        self.hyperSize     = hyperSize
        self.expMaxSize    = 2**(self.hyperSize)-1 # the maximum value that can be represented by self.hyperSize bits, using standard binary representation. 
        if (self.hyperSize + self.expMaxSize > self.cntrSize-1):
            print ('Requested hyperSize {} is not feasible for counter size {}' .format (hyperSize, self.cntrSize))
            return False
        return True

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
        
