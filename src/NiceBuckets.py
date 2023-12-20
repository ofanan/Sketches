# Buckets of Counter arrays.
import math, random, os, pickle, mmh3, time
import numpy as np
from datetime import datetime

import settings, Buckets, IceBucket
from printf import printf, printarFp

class CntrMaster (Buckets.Buckets):
    """
    New (improved) IceBuckets.
    """

    def queryCntrVal (self, cntrIdx=0):
        """
        Query a cntr. 
        Input: cntrIdx - the counter's index. 
        Output: The value that the counter represents (as int/FP).
        """
        settings.error ('Sorry. NiceBuckets.queryCntrVal() is not implemented yet.')
        # return self.regBkts[self.idx2BucketNum(cntrIdx)].queryCntrVal(cntrIdx=cntrIdx%self.numCntrsPerBkt)
            
    def __init__ (self, 
                  cntrSize          = 4, # num of bits in each counter. 
                  numCntrs          = 9, # number of counters in the array.
                  numCntrsPerBkt    = 1, # number of cntrs at each bucket.
                  numCntrsInXlBkt   = 1,
                  numEpsilonSteps   = 8,
                  numEpsilonStepsInXlBkt = 4,
                  verbose           = [], # determines which outputs would be written to .log/.res/.pcl/debug files, as detailed in settings.py.
                  ):

        if cntrSize<1 or numCntrs<1:
            settings.error (f'in Buckets: you requested cntrSize={cntrSize}, numCntrs={numCntrs}. However, you should choose cntrSize>=1, numCntrs>=1.')
            
        self.cntrSize, self.numCntrs, self.numCntrsPerBkt = int(cntrSize), int(numCntrs), int(numCntrsPerBkt)
        self.numCntrsInXlBkt = numCntrsInXlBkt
        self.numRegularBuckets = self.numCntrs // self.numCntrsPerBkt
        self.verbose    = verbose
        # self.minValOfXlBkt = 
        self.regBkts = [IceBucket.CntrMaster(
                            cntrSize        = self.cntrSize, 
                            numCntrs        = self.numCntrsPerBkt,
                            numEpsilonSteps = numEpsilonSteps,
                            verbose         = self.verbose)
                            for _ in range (self.numRegularBuckets)]        
        self.xlBkt = IceBucket.CntrMaster(
                            cntrSize        = self.cntrSize, 
                            numCntrs        = self.numCntrsInXlBkt,
                            numEpsilonSteps = numEpsilonStepsInXlBkt,
                            verbose         = self.verbose)
        
    def printAllCntrs (self, outputFile) -> None:
        """
        Format-print all the counters as a single the array, to the given file.
        """
        printf (outputFile, '[')
        for bkt in self.regBkts:
            bkt.printAllCntrVals(outputFile)
        printf (outputFile, ']')
    
    def rstCntr (self, cntrIdx=0) -> None:
        """
        Reset a single counter.
        """
        self.regBkts[self.idx2BucketNum(cntrIdx)].rstCntr(idx%self.numCntrsPerBkt)
        # self.xlBkt.rstCntr(idx%self.numCntrsPerBkt)
    
    def queryCntr (self, cntrIdx=0) -> dict:
        """
        Query a cntr.
        Input: 
        cntrIdx - the counter's index. 
        Output:
        cntrDic: a dictionary, where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.        
        """
        settings.error ('Sorry. NiceBuckets.queryCntr() is not implemented yet.')
        # return self.regBkts[self.idx2BucketNum(cntrIdx)].cntr2cntrDict(cntrIdx%self.numCntrsPerBkt)           

    def incCntrBy1GetVal (self, 
                          cntrIdx  = 0, # idx of the concrete counter to increment in the array 
                          ) -> int: # If forceInc==True, increment the counter. Else, inc the counter w.p. p 
        """
        Increment the counter to the closest higher value.        
        Output:
        cntrDict: a dictionary representing the modified counter where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.
        """
        settings.error ('Sorry, NiceBuckets.incCntrBy1GetVal() is not implemented yet')
        # return self.regBkts[self.idx2BucketNum(cntrIdx)].incCntrBy1GetVal (cntrIdx=cntrIdx%self.numCntrsPerBkt)
 

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
            settings.error ('Sorry, Buckets.incCntr() is currently implemented only when mult==True and factor=1.')
        return self.regBkts[self.idx2BucketNum(cntrIdx)].incCntrBy1 (cntrIdx=cntrIdx%self.numCntrsPerBkt) 

    def setLogFile (self, logFile):
        """
        set the log file
        """ 
        self.logFile = logFile
        for bkt in self.regBkts:
            bkt.logFile = logFile

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
            settings.error ('Sorry, NiceBuckets.incCntrGetVal() is currently implemented only when mult==True and factor=1.')

