""" 
General class of counters, with some functionalities common to all the counters.
""" 
import math, time, random, numpy as np

import settings, Cntr
from settings import VERBOSE_DEBUG, error
from printf import printf

class CntrMaster (object):
    
    # Return a range with all the legal combinations for the counter. For most counters this includes all the possible binary comb's.  
    getAllCombinations = lambda self, cntrSize : range (2**cntrSize)
    
    def __init__ (self, 
                  cntrSize=4,   # num of bits in each counter.
                  numCntrs=1,   # number of counters in the array. 
                  verbose=[]    # one of the verbose macros, detailed in settings.py
                  ):
        """
        Initialize an array of cntrSize counters. The cntrs are initialized to 0.
        """
        
        if (cntrSize<3):
            error ('cntrSize requested is {}. However, cntrSize should be at least 3.' .format (cntrSize))
        self.cntrSize   = int(cntrSize)
        self.numCntrs   = int(numCntrs)
        self.verbose    = verbose
        
    def printAllCntrs (
            self, 
            outputFile,
            printAlsoVec = False, # when True, print also the counters' vectors.
            printAsInt   = False  # when True, print the value as an integer 
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
                    print (f'{self.queryCntr(cntrIdx=idx, getVal=True)} ')
        else:
            cntrVals = np.empty (self.numCntrs)
            for idx in range(self.numCntrs):
                cntrVals[idx] = self.queryCntr(cntrIdx=idx, getVal=True)
            printf (outputFile, 
                    '// minCntrVal={:.1f}, maxCntrVal={:.1f}, avgCntrVal={:.1f} \n// cntrsVals:\n'
                    .format (np.min(cntrVals), np.max(cntrVals), np.average(cntrVals)))
            if printAsInt:
                for cntrVal in cntrVals:
                    printf (outputFile, '{:.0f} ' .format(cntrVal))
            else:
                for cntrVal in cntrVals:
                    printf (outputFile, f'{cntrVal} ')


    def printCntrsStat (
            self, 
            outputFile, # file to which the stat will be written
            genPlot         = False, # when True, plot the stat 
            outputFileName  = None, # filename to which the .pdf plot will be saved
        ) -> None:
        """
        An empty function. Implemented only for compatibility with buckets, that do have such a func.
        """
        None

    def setLogFile (self, logFile):
        """
        An empty function. Implemented only for compatibility with buckets, that do have such a func.
        """
        None 
    
    
    def rstCntr (self, cntrIdx=0):
        """
        """
        self.cntrs[cntrIdx] = self.cntrZeroVec
        
    
    def queryCntr (self, 
            cntrIdx  = 0, #  
            getVal   = True # If True, return only the counter's value. Else, return cntrDic - a dictionary, where cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.
        ):
        """
        Query a cntr.
        Input: 
         
        Output:
        cntrDic: a dictionary, where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.        
        """
        if getVal:
            return self.cntr2num(self.cntrs[cntrIdx])
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='SEAD')
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : self.cntr2num(self.cntrs[cntrIdx])}    

    def incCntr (self, cntrIdx=0, factor=int(1), mult=False, verbose=None):
        """
        """
        if verbose!=None:
            self.verbose = verbose
        if factor==1 and mult==False:
            return self.incCntrBy1GetVal (cntrIdx)
    
        settings ('Sorry. Cntr.incCntr() is currently implemented only as incCntrBy1.')
    
    def getAllVals (self, verbose=[]):
        """
        Loop over all the binary combinations of the given counter size. 
        For each combination, calculate the respective counter, and its value. 
        Returns a vector of these values, sorted in an increasing order of the counters' values. 
        """
        listOfVals = []
        for i in self.getAllCombinations (self.cntrSize):
            cntr = np.binary_repr(i, self.cntrSize) 
            listOfVals.append ({'cntrVec' : cntr, 'val' : self.cntr2num(cntr)})
        listOfVals = sorted (listOfVals, key=lambda item : item['val'])
    
        if settings.VERBOSE_RES in verbose:
            outputFile    = open ('../res/log_files/{}.res' .format (self.genSettingsStr()), 'w')
            for item in listOfVals:
                printf (outputFile, '{}={}\n' .format (item['cntrVec'], item['val']))
        return [item['val'] for item in listOfVals]

    def rstAllCntrs (self):
        """
        """
        self.cntrs = [self.cntrZeroVec]*self.numCntrs

        
    def getCntrMaxVal (self):
        return self.cntrMaxVal        
    