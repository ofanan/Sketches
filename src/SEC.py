# An array (single bucket) of Shared Exponent Counters (SEC).
# Each counter contains the Mantissa.
# A bucket is composed of the counters, and a single, shared, exponent value. 
import math, time, random
import numpy as np
from printf import printf, printarFp
import settings

class CntrMaster (object):
    """
    Generate, check and parse counters
    """

    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr  = lambda self : f'SEC_n{self.cntrSize}_e{self.exp}'
       
    # Given a counter, return the number it represents
    cntr2num        = lambda self, cntr : cntr << self.exp

    # Given a counter, return a dictionary cntrDict, where: cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.            
    cntr2cntrDict   = lambda self, cntr : {'cntrVec' : np.binary_repr(cntr, width=self.cntrSize), 'val' : self.cntr2num(cntr)}
    
    # Query a cntr. 
    # Input: cntrIdx - the counter's index. 
    # Output: The value that the counter represents (as int/FP).
    queryCntrVal    = lambda self, cntrIdx=0 : self.cntr2num(self.cntrs[cntrIdx])              
        
    def __init__ (self, 
                  cntrSize      = 4, # num of bits in each counter. 
                  numCntrs      = 2, # number of counters in the array.
                  verbose       = [], # determines which outputs would be written to .log/.res/.pcl/debug files, as detailed in settings.py. 
                  useOnlyFloorDivision = False, # If True, each time a division by 2 is performed, use floor division. When False, choose either floor or ceil with prob' 0.5 each.
                  ):
        

        if cntrSize<1 or numCntrs<1:
            settings.error (f'in SEC: you requested cntrSize={cntrSize}, numCntrs={numCntrs}. However, cntrSize and numCntrs should be at least 1.')
            
        self.cntrSize, self.numCntrs = int(cntrSize), int(numCntrs)
        self.useOnlyFloorDivision    = useOnlyFloorDivision
        self.verbose                 = verbose
        self.cntrMaxMantissaVal      = (1 << self.cntrSize) - 1
        self.rst () # reset all the counters
        
    def printCntrs (self, outputFile) -> None:
        """
        Format-print all the counters as a single the array, to the given file.
        """
        if outputFile==None:
            print (f'Printing all cntrs. exp={self.exp}')
            for cntr in self.cntrs:
                print (f'{self.cntr2num(cntr)} ')
        else:
            for cntr in self.cntrs:
                printf (outputFile, f'{self.cntr2num(cntr)} ')
    
    def rstCntr (self, cntrIdx=0):
        """
        Reset a single counter.
        """
        self.cntrs[cntrIdx] = [0]*self.numCntrs

    def rst (self):
        """
        Reset all the counters.
        """
        if self.cntrSize<=8:
            self.cntrs      = np.zeros(self.numCntrs, dtype='uint8') 
        elif self.cntrSize<=16:
            self.cntrs      = np.zeros(self.numCntrs, dtype='uint16') 
        else:
            self.cntrs      = np.zeros(self.numCntrs, dtype='uint32')             
        self.exp        = 0
        self.sampleProb = 1

    def queryCntr (self, cntrIdx=0) -> dict:
        """
        Query a cntr.
        Input: 
        cntrIdx - the counter's index. 
        Output:
        cntrDic: a dictionary, where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.        
        """
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='SEC')
        return self.cntr2cntrDict(self.cntrs[cntrIdx])           

    def incExp (self):
        """
        Increment the exponent by 1 and divide all the counters by 2.
        """
        self.exp        += 1
        self.sampleProb /= 2 
        if self.useOnlyFloorDivision:
            self.cntrs = [cntr//2 for cntr in self.cntrs]
        else:
            self.cntrs = [(math.floor(cntr/2) if random.random()<0.5 else math.ceil(cntr/2)) for cntr in self.cntrs]

    def incCntrBy1GetVal (self, 
                          cntrIdx  = 0, # idx of the concrete counter to increment in the array 
                          ) -> int: # If forceInc==True, increment the counter. Else, inc the counter w.p. p 
        """
        Increment the counter to the closest higher value.        
        Output:
        cntrDict: a dictionary representing the modified counter where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.
        """
        if self.exp==0 or (random.random() < self.sampleProb):
            if (self.cntrs[cntrIdx]==self.cntrMaxMantissaVal):
                self.cntrs[cntrIdx] += 1
                if settings.VERBOSE_LOG in self.verbose:
                    printf (self.logFile, 'inc exp\n')
                self.incExp()
            else:
                self.cntrs[cntrIdx] += 1
        return self.cntr2num(self.cntrs[cntrIdx])

    def incCntr (self, cntrIdx=0, factor=1, verbose=[], mult=False):
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
        """
        if mult or (factor!=1):
            settings.error ('Sorry, SEC.incCntr() is currently implemented only when mult==True and factor=1.')
        return self.incCntrBy1 (cntrIdx=cntrIdx) 

    def incCntrGetVal (self, cntrIdx=0, factor=1, verbose=[], mult=False):
        """
        Increase a single counter by a given factor.
        Input:
        cntrIdx - index of the cntr to increment, in the array of cntrs.
        mult - if true, multiply the counter by factor. Else, increase the counter by factor.
        factor - the additive/multiplicative coefficient.
        verbose - determines which data will be written to the screen.
        Output:
        The value of the modified counter.
        Operation:
        Define cntrVal as the current counter's value. 
        Then, targetValue = cntrVal*factor (if mult==True), and targetValue = cntrVal + factor (otherwise).  
        If targetValue > maximum cntr's value, return the a cntr representing the max possible value. 
        If targetValue < 0, return a cntr representing 0.
        If targetValue can be represented correctly by the counter, return the exact representation.
        Else, use probabilistic cntr's modification.
        """
        if mult or (factor!=1):
            settings.error ('Sorry, SEC.incCntrGetVal() is currently implemented only when mult==True and factor=1.')
        return self.incCntrBy1GetVal (cntrIdx=cntrIdx) 

def printAllVals (cntrSize=4, verbose=[], cntMax=100):
    """
    Loop over all the binary combinations of the given counter size. 
    For each combination, print to file the respective counter, and its value. 
    The prints are sorted in an increasing order of values.
    """
    myCntrMaster = CntrMaster(cntrSize=cntrSize)

    if (settings.VERBOSE_RES in verbose):
        outputFile    = open ('../res/{}.res' .format (myCntrMaster.genSettingsStr()), 'w')
    
    print ('running printAllVals')
    listOfVals = [0]
    for i in range (cntMax):
        listOfVals.append (myCntrMaster.incCntrBy1GetVal())
    
    if (settings.VERBOSE_RES in verbose):
        for item in listOfVals:
            printf (outputFile, f'{item}\n')
    
    if (settings.VERBOSE_PCL in verbose):
        with open('../res/pcl_files/{}.pcl' .format (myCntrMaster.genSettingsStr()), 'wb') as pclOutputFile:
            pickle.dump(listOfVals, pclOutputFile) 
      
# printAllVals(cntrSize=4, verbose=[settings.VERBOSE_RES])
