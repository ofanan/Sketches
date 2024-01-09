# A CounterMaster that manages an array of perfect counters.
# Each perfect counter is an integer counter that can accurately count any integer value.

import numpy as np
from printf import printf, printarFp

class CntrMaster(object):

    genSettingsStr = lambda self : ''

    def __init__(self, cntrSize, numCntrs, verbose=[]):
        """
        first i have initialized  all the counters which has cntrSize bits to zero. Eg '000000' if cntrSize is 6.
        As count min sketch is dimensional array with rows equals to number of depth and columns equals to number of width,
         i have converted it to row and column pair list.
        """
        self.numCntrs   = numCntrs  # number of counters in the flow array, which is width*depth
        self.verbose    = verbose
        self.rstAllCntrs()

    def setLogFile (self, logFile):
        """
        set the log file
        """ 
        self.logFile = logFile

    def printAllCntrs (self, outputFile) -> None:
        """
        Format-print all the counters as a single the array, to the given file.
        """
        printf (outputFile, '\n[')
        for cntr in self.cntrs:
            printf (outputFile, f'{cntr} ')
        printf (outputFile, ']')
    
    def rstAllCntrs(self):
        """
        """
        self.cntrs = [0 for i in range(self.numCntrs)]
    def rstCntr (self, cntrIdx=0):
        """
        """
        self.cntrs[cntrIdx] = 0

    def incCntrBy1GetVal(self, cntrIdx=0):
        """

        This converts the counter binary value to integer and check if that value can increment or reaches its max value. If it not reaches max
        value, it added 1 to the target value and save it as binary.
        """
        self.cntrs[cntrIdx] += 1
        return self.cntrs[cntrIdx]

    def incCntrGetVal(self, cntrIdx=0, factor=1, mult=False):
        """

        This converts the counter binary value to integer and check if that value can increment or reaches its max value. If it not reaches max
        value, it added 1 to the target value and save it as binary.
        """
        if mult:
            self.cntrs[cntrIdx] *= factor
        else:
            self.cntrs[cntrIdx] += factor
        return self.cntrs[cntrIdx]

    def queryCntr(self, cntrIdx):
        """

        Here i used the variable flowIdx to get the binary number from counters list, and converted it to number
        """
        return self.cntrs[cntrIdx]