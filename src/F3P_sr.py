# This file implements F3P SR, namely, F2P flavor that focuses on improved accuracy on small reals. 
# For further details, see "main.tex" in Cntr's Overleaf project.
import math, random, pickle, numpy as np

import settings, Cntr, F2P_sr
from settings import *
from printf import printf

class CntrMaster (F2P_sr.CntrMaster):
    """
    Generate, check and perform arithmetic operations of the counters. 
    The counters are generated as an array with the same format (counterSize and hyperExpSize).
    Then, it's possible to perform arithmetic ops on the counters in chosen indices of the array. 
    """

    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr = lambda self : f'F3P_{self.flavor()}_n{self.cntrSize}_h{self.hyperMaxSize}'
    
    # Given the vector of the exponent, calculate the value it represents 
    expVec2expVal  = lambda self, expVec, expSize : (2**expSize - 1 + int (expVec, base=2)) if expSize>0 else 0    
    
    def setFlavorParams (self):
        """
        set variables that are unique for 'sr' flavor of F2P.
        """
        super().setFlavorParams()
        self.cntrMaxVec = '1'*self.cntrSize #  np.binary_repr  (2**self.cntrSize-1, self.cntrSize) # the cntr that reaches the highest value is "11...11"
        
    def calcParams (self):
        """
        Calc the basics param, which are depended upon the counter size, and the hyper-exp' size.
        """
        self.mantMinSize = self.cntrSize - self.hyperMaxSize - self.expMaxSize 
        if (self.mantMinSize<1):
            error (f'In F3P_sr.calcParams(). cntrSize={self.cntrSize} and hyperMaxSize={self.hyperMaxSize} implies min mantissa size={self.mantMinSize}. Mantissa size should be at least 1. Please use a smaller hyperMaxSize')
        self.Vmax = 2**(self.expMaxSize+1)-1 
        self.setFlavorParams ()
        self.powerMin = self.expMinVal + self.bias + 1 
   
    def __init__ (self, 
      cntrSize      : int = 8, # of bits in the cntr 
      hyperMaxSize  : int = 1, # of bits in the hyper-exp field 
      numCntrs      : int = 1, # of cntrs in the cntrs' array
      verbose             = []    # the optional verbose values are detailed in settings.py
    ):
        
        """
        Initialize an array of cntrSize counters. The cntrs are initialized to 0.
        If the parameters are invalid (e.g., infeasible cntrSize), return None. 
        """
        if (cntrSize<3):
            error ('In F3P_sr.__init(): cntrSize requested is {}. However, cntrSize should be at least 3.' .format (cntrSize))
        self.cntrSize   = int(cntrSize)
        self.numCntrs   = int(numCntrs)
        self.verbose    = verbose
        self.setHyperSize (hyperMaxSize) # set the field maxHyperSize and check whether the configuration is valid. 
        self.calcParams   () # set some parameters and check whether the configuration is valid.
        # self.cntrMinVal = self.cntr2num (self.cntrZeroVec)
        self.cntrMaxVal = self.cntr2num (self.cntrMaxVec)
        if VERBOSE_COUT_CONF in self.verbose:
            print (f'F3P{self.flavor()}, cntrSize={self.cntrSize}, hyperMaxSize={self.hyperMaxSize}, bias={self.bias}, Vmax={self.Vmax}, bias={self.bias}, zeroVec={self.cntrZeroVec}, maxVec={self.cntrMaxVec}, maxVal={self.cntrMaxVal}, expMinVec={self.expMinVec}, expMinVal={self.expMinVal}')
        self.rstAllCntrs ()
        self.isFeasible = True
        
    def cntr2num (self, cntr):
        """
        Given a counter, as a binary vector (e.g., "11110"), return the number it represents.
        """
        if (len(cntr) != self.cntrSize): # if the cntr's size differs from the default, we have to update the basic params
            error (f'In F3P_{self.flavor()}.cntr2num(). the size of the given counter {cntr} is {len(cntr)} while CntrMaster was initialized with cntrSize={self.cntrSize}')

        # Extract the hyper-exponent field, and value
        self.hyperSize = settings.idxOfLeftmostZero (ar=cntr, maxIdx=self.hyperMaxSize)         
        expSize      = self.hyperSize
        if self.hyperSize < self.hyperMaxSize: # if the # of trailing max < hyperMaxSize, the cntr must have a delimiter '0'
            expVecBegin  = self.hyperSize+1
        else:
            expVecBegin  = self.hyperMaxSize
        expVec  = cntr[expVecBegin : expVecBegin+expSize]
        mantVec = cntr[expVecBegin+expSize:]
        expVal  = self.expVec2expVal(expVec, expSize) 
        mantVal = float (int (mantVec, base=2)) / 2**(self.cntrSize - expVecBegin - expSize)  
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
            self.printCntrLine (
                cntr    = cntr, 
                expVec  = expVec, 
                expVal  = expVal, 
                power   = power, 
                mantVec = mantVec, 
                mantVal = mantVal, 
                cntrVal = cntrVal)
        return cntrVal
    
    def setHyperSize (self, hyperMaxSize):
        """ 
        Sets the size of the hyper-exponent field in F2P counters as follows.
        - Check whether the hyper-exponent field size is feasible.
        - If yes - assign the relevant "self" fields (exponent's field max-size). return True
        - If not - finish with an error msg.
        """
        if (hyperMaxSize<1 or hyperMaxSize>self.cntrSize-2):
            error (f'In F3P_sr.setHyperSize(): Requested hyperMaxSize {hyperMaxSize} is not feasible for counter size {self.cntrSize}')
        self.hyperMaxSize  = hyperMaxSize
        self.expMaxSize    = self.hyperMaxSize # the maximum value that can be represented by self.hyperMaxSize bits, using standard binary representation. 
        if (self.hyperMaxSize + self.expMaxSize > self.cntrSize-1):
            error (f'In F3P_sr.setHyperSize(): Requested hyperMaxSize {hyperMaxSize} is not feasible for counter size {self.cntrSize}')

