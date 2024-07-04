# This file implements F3P LI_ds, namely, F3P flavor that represents only integers, and focuses on improved accuracy on elephants; and supports down-sampling.
# This class mainly inherits from the class F3P_li. 
# For further details, see "main.tex" in Cntr's Overleaf project.
import math, random, pickle, numpy as np

from printf import printf
import settings, F3P_li
from settings import error, VERBOSE_DEBUG, VERBOSE_LOG_DWN_SMPL

class CntrMaster (F3P_li.CntrMaster):
    """
    Generate, check and perform arithmetic operations on F3P counters in LI flavors with down-sampling.
    The counters are generated as an array with the same format (counterSize and hyperExpSize).
    Then, it's possible to perform arithmetic ops on the counters in chosen indices of the array. 
    """

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
        self.globalIncProb = 1.0 # Probability to consider an increment for any counter. After up-scaling, this probability decreases.
        super(CntrMaster, self).__init__ (
            cntrSize    = cntrSize, 
            numCntrs    = numCntrs,
            hyperSize   = hyperSize, 
            verbose     = verbose
        )

    def setFlavorParams (self):
        """
        set variables that are unique for 'li' flavor of F3P.
        """
        super(CntrMaster, self).setFlavorParams ()
        self.LsbVecOfAbsExpVal   = ['']*(self.Vmax) # self.LsbVecOfAbsExpVal[e] will hold the LSB fields (hyperVec and expVec) of the vector when decreasing the vector's exponent whose current absolute value is e.
        for expSize in range(self.expMaxSize, 0, -1):
            hyperVec = np.binary_repr (expSize, self.hyperSize) 
            mantSize = self.cntrSize - self.hyperSize - expSize
            for i in range (2**expSize-1, 0, -1): 
                expVec = np.binary_repr(num=i, width=expSize)
                expVal = self.expVec2expVal (expVec=expVec, expSize=expSize)
                self.LsbVecOfAbsExpVal[abs(expVal)-1] = hyperVec + expVec 
            expVal = self.expVec2expVal (expVec='0'*expSize, expSize=expSize)
            self.LsbVecOfAbsExpVal[abs(expVal)-1] = hyperVec + '0'*expSize
        self.LsbVecOfAbsExpVal[self.Vmax-1] = '1'*(self.hyperSize + 2**self.hyperSize - 1) 
        self.mantSizeOfAbsExpVal = [self.cntrSize - len(item) for item in self.LsbVecOfAbsExpVal] # self.mantSizeOfAbsExpVal[e] is the size of the mantissa field of the vector when decreasing the vector's exponent whose current absolute value is e.
        
    def printAllCntrs (
            self, 
            outputFile   = None,
            printAlsoVec = False, # when True, print also the counters' vectors.
        ) -> None:
        """
        Format-print all the counters as a single the array, to the given file.
        Format print the values corresponding to all the counters in self.cntrs.
        Used for debugging/logging.
        """        
        if outputFile==None:
            print (f'Printing all cntrs.')
            if printAlsoVec:
                for idx in range(self.numCntrs):
                    cntrDict = self.queryCntr (idx, getVal=False)
                    print ('cntrVec={}, cntrVal={} ' .format (cntrDict['cntrVec'], cntrDict['val']))
            else:
                for idx in range(self.numCntrs):
                    print ('{:.0f} ' .format(self.queryCntr(cntrIdx=idx, getVal=True)))
        else:
            cntrVals = np.empty (self.numCntrs)
            for idx in range(self.numCntrs):
                cntrVals[idx] = self.queryCntr(cntrIdx=idx, getVal=True)
            printf (outputFile, 
                    '// minCntrVal={:.0f}, maxCntrVal={:.0f}, avgCntrVal={:.0f} \n// cntrsVals:\n'
                    .format (np.min(cntrVals), np.max(cntrVals), np.average(cntrVals)))
            for cntrVal in cntrVals:
                printf (outputFile, '{:.0f} ' .format(cntrVal))

    def setDwnSmpl (
            self, 
            dwnSmpl   : bool = False, # When True, use down-sampling 
        ):
        
        """
        """
        self.dwnSmpl = dwnSmpl
        error ('In F3P_li.setDwnSmpl(). dwnSmpl is not implemented yet for F3P_li.')

    def incCntr (self, cntrIdx=0, factor=int(1), mult=False, verbose=[]):
        """
        Increment the counter to the closest higher value.
        If the cntr reached its max val, or the randomization decides not to inc, merely return the cur cntr.
        Return the counter's value after the increment        
        """
        if factor==1 and mult==False:
            return self.incCntrBy1GetVal (cntrIdx=cntrIdx)
        settings.error ('In F3P_li.incCntr(). Sorry, incCntr is currently supported only for factor=1 and mult=False')
    
    def incCntrBy1GetVal (self, 
                    cntrIdx  = 0, # idx of the concrete counter to increment in the array
                    forceInc = False # If forceInc==True, increment the counter. Else, inc the counter w.p. corresponding to the next counted value.
                    ): 
        """
        Increment the counter to the closest higher value.
        If the cntr reached its max val, or the randomization decides not to inc, merely return the cur cntr.
        Return the counter's value after the increment        
        """
        
        cntr        = self.cntrs[cntrIdx]
        hyperSize   = settings.idxOfLeftmostZero (ar=cntr, maxIdx=self.hyperMaxSize)
        if hyperSize==self.hyperMaxSize: # need no delimiter
            expVec = cntr[hyperSize:2*hyperSize]
        else:
            expVec = cntr[(hyperSize+1):(2*hyperSize+1)]
        expVal     = int (self.expVec2expVal(expVec=expVec, expSize=hyperSize))
        mantSize   = self.cntrSize - 2*hyperSize
        if hyperSize<self.hyperMaxSize:
            mantSize -=1 # recall 1 more bit for the delimiter. 
        mantVec    = cntr[-mantSize:]
        mantIntVal = int (mantVec, base=2)
        mantVal    = float (mantIntVal) / 2**mantSize  

        if expVec == self.expMinVec:
            cntrCurVal = mantVal * (2**self.powerMin)
        else:
            cntrCurVal = (1 + mantVal) * (2**(expVal+self.bias))
        
        if not(forceInc): # check first the case where we don't have to inc the counter 
            if (self.cntrs[cntrIdx]==self.cntrMaxVec or random.random() > self.probOfInc1[abs(expVal)]): 
                return int(cntrCurVal)    

        # now we know that we have to inc. the cntr
        cntrppVal  = cntrCurVal + (1/self.probOfInc1[abs(expVal)])
        if settings.VERBOSE_COUT_CNTRLINE in self.verbose:
            print (f'b4 inc: cntrVec={cntr}, cntrVal={int(cntrCurVal)}')
        if mantVec == '1'*mantSize: # the mantissa overflowed
            self.cntrs[cntrIdx] = self.cntrppOfAbsExpVal[abs(expVal)]
        else:
            if hyperSize<self.hyperMaxSize:
                self.cntrs[cntrIdx] = '1'*hyperSize + '0' + expVec + np.binary_repr(num=mantIntVal+1, width=mantSize) 
            else:
                self.cntrs[cntrIdx] = '1'*hyperSize       + expVec + np.binary_repr(num=mantIntVal+1, width=mantSize) 
        if settings.VERBOSE_COUT_CNTRLINE in self.verbose:
            print (f'after inc: cntrVec={self.cntrs[cntrIdx]}, cntrVal={int(cntrppVal)}')
        return int(cntrppVal) 
        