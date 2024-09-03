""" 
General class of counters, with some functionalities common to all the counters.
""" 
import math, time, random, numpy as np

import settings, Cntr
from settings import *
from printf import printf

class CntrMaster (object):
    
    # Generates a string that details the counter's settings (param vals).
    genSettingsStr = lambda self : f'cntr_n{self.cntrSize}'
    
    # Return a range with all the legal combinations for the counter. For most counters this includes all the possible binary comb's.  
    getAllCombinations = lambda self, cntrSize : range (2**cntrSize)
    
    def __init__ (self, 
        cntrSize    = 4,   # num of bits in each counter.
        numCntrs    = 1,   # number of counters in the array. 
        verbose     = []    # one of the verbose macros, detailed in settings.py
    ):
        """
        Initialize an array of cntrSize counters. The cntrs are initialized to 0.
        """
        
        if (cntrSize<3):
            error ('In Cntr.__init(). cntrSize requested is {}. However, cntrSize should be at least 3.' .format (cntrSize))
        self.cntrSize       = int(cntrSize)
        self.numCntrs       = int(numCntrs)
        self.verbose        = verbose
        self.allowDwnSmpl   = False # Default; down-sampling is allowed only for some concrete child classes, that set this parameter. 
        
    def rmvVerbose (
            self, 
            verbose : int, # verbose to remove; verboses are defined in settings.py
        ):
        """
        Rmv the requested verbose codes from self.verbose. 
        Typically called after the first experiment, as no need to log more than a single exp.
        """
        if verbose in self.verbose:
            self.verbose.remove(verbose)
        
    
    def printAllCntrs (
            self, 
            outputFile   = None,
            printAlsoVec = True, # when True, print also the counters' vectors.
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
                    cntrDict = self.queryCntr (idx, getVal=True)
                    cntrVal  = cntrDict['val']
                    if cntrVal==int(cntrVal):
                        print ('cntrVec={}, cntrVal={} ' .format (cntrDict['cntrVec'], cntrVal))
                    else:
                        print ('cntrVec={}, cntrVal={:.1f} ' .format (cntrDict['cntrVec'], cntrVal))
            else:
                for idx in range(self.numCntrs):
                    print ('{} ' .format(self.cntr2num(cntr=self.cntrs[idx])))
        else:
            printf (outputFile, 'cntrs=[')
            if printAlsoVec:
                for idx in range(self.numCntrs):
                    cntrDict = self.queryCntr (idx, getVal=False)
                    cntrVal  = cntrDict['val']
                    if cntrVal==int(cntrVal):
                        printf (outputFile, 'cntrVec={}, cntrVal={} ' .format (cntrDict['cntrVec'], cntrVal))
                    else:
                        printf (outputFile, 'cntrVec={}, cntrVal={:.1f} ' .format (cntrDict['cntrVec'], cntrVal))
                printf (outputFile, '] ')
            else:
                for idx in range(self.numCntrs):
                    printf (outputFile, '{} ' .format(self.cntr2num(self.cntrs[idx])))
                printf (outputFile, '] ')


    def printCntrsStat (
            self, 
            outputFile, # file to which the stat will be written
            genPlot         = False, # when True, plot the stat 
        ) -> None:
        """
        An empty function. Implemented only for compatibility with buckets, that do have such a func.
        """
        settings.writeVecStatToFile (
            statFile    = outputFile,
            vec         = [self.cntr2num(cntr) for cntr in self.cntrs],
            str         = 'cntrs'        
        )
        if self.numCntrs<10:
            self.printAllCntrs(
                outputFile      = outputFile, 
                printAlsoVec    = False
            )

    def setLogFile (self, logFile):
        """
        An empty function. Implemented only for compatibility with buckets, that do have such a func.
        """
        self.logFile = logFile 
    
    
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

    def setDwnSmpl (
            self, 
            dwnSmpl   : bool = False, # When True, use down-sampling 
        ):        
        """
        Set the down-sampling for relevant cntr's types (child classes).
        By default, this feature is not supported, and therefore the run ends with an error message.
        """
        error ('In Cntr.setDwnSmpl(). Sorry. Down sampling is not yet implemented for self.genSettingsStr()')

    def dwnSmpl (
            self
        ):
        """
        down-sample.
        By default, this feature is not supported, and therefore the run ends with an error message.
        """
        error ('In Cntr.dwnSmpl(). Sorry. Down sampling is not yet implemented for self.genSettingsStr()')

    def rstAllCntrs (self):
        """
        """
        self.cntrs = [self.cntrZeroVec]*self.numCntrs

        
    def getCntrMaxVal (self):
        return self.cntrMaxVal        
    