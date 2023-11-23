# Buckets of Counter arrays.
import math, random, os, pickle, mmh3, time
import numpy as np
from datetime import datetime

import settings, SEC, CEDAR
from printf import printf, printarFp

class Buckets (object):
    """
    Generate, check and parse counters
    """

    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr  = lambda self : f'bkts{self.numBuckets}'

    # Given the index in the Buckets, get the bucket number and the number of the counter in the bucket
    getBucketNumAndCntrNum = lambda self, idx : [self.idx2BucketNum(idx), idx%self.numCntrsPerBkt]
       
    # Given the index in the Buckets, get the bucket number 
    idx2BucketNum = lambda self, idx : idx//self.numCntrsPerBkt
    
    # Query a cntr. 
    # Input: cntrIdx - the counter's index. 
    # Output: The value that the counter represents (as int/FP).
    queryCntrVal    = lambda self, cntrIdx=0 : self.buckets[self.idx2BucketNum(cntrIdx)].queryCntrVal(cntrIdx=cntrIdx%self.numCntrsPerBkt)                  
            
    def __init__ (self, 
                  cntrSize          = 4, # num of bits in each counter. 
                  numCntrs          = 9, # number of counters in the array.
                  cntrMaxVal        = 10, 
                  numCntrsPerBkt    = 1, # number of cntrs at each bucket.
                  mode              = 'SEC',
                  EStep             = 0.1,
                  numESteps         = 8,
                  initialEpsilon    = 0.1,  # initial value of the epsilon accuracy parameter, defined at the paper ICE_buckets.
                  verbose           = [], # determines which outputs would be written to .log/.res/.pcl/debug files, as detailed in settings.py.
                  ):

        if cntrSize<1 or numCntrs<1:
            settings.error (f'in Buckets: you requested cntrSize={cntrSize}, numCntrs={numCntrs}. However, you should choose cntrSize>=1, numCntrs>=1.')
            
        self.cntrSize, self.numCntrs, self.numCntrsPerBkt = int(cntrSize), int(numCntrs), int(numCntrsPerBkt)
        self.numBuckets = self.numCntrs // self.numCntrsPerBkt
        self.verbose    = verbose
        self.mode       = mode
        if mode=='SEC':
            self.buckets = [SEC.CntrMaster(cntrSize=self.cntrSize, numCntrs=self.numCntrsPerBkt, verbose=self.verbose) for _ in range (self.numBuckets)]
        elif mode=='CEDAR':
            self.buckets = [CEDAR.CntrMaster(
                                            cntrSize        = self.cntrSize, 
                                            numCntrs        = self.numCntrsPerBkt,
                                            cntrMaxVal      = cntrMaxVal, 
                                            EStep           = EStep,
                                            numESteps       = numESteps,
                                            initialEpsilon  = initialEpsilon,  # initial value of the epsilon accuracy parameter, defined at the paper ICE_buckets.
                                            verbose=self.verbose) for _ in range (self.numBuckets)]
        else:
            settings.error ('Sorry. Mode {self.mode} that you chose is not supported yet by Buckets.py.')
        
        
    def printAllCntrs (self, outputFile) -> None:
        """
        Format-print all the counters as a single the array, to the given file.
        """
        printf (outputFile, '[')
        for bkt in self.buckets:
            bkt.printCntrs(outputFile)
        printf (outputFile, ']')
    
    def rstCntr (self, cntrIdx=0) -> None:
        """
        Reset a single counter.
        """
        self.buckets[self.idx2BucketNum(cntrIdx)].rstCntr(idx%self.numCntrsPerBkt)
    
    # def rst (self, cntrIdx=0):
    #     """
    #     Reset all the counters.
    #     """
    #     for bkt in self.buckets:
    #         bkt.rst ()

    def queryCntr (self, cntrIdx=0) -> dict:
        """
        Query a cntr.
        Input: 
        cntrIdx - the counter's index. 
        Output:
        cntrDic: a dictionary, where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.        
        """
        return self.buckets[self.idx2BucketNum(cntrIdx)].cntr2cntrDict(cntrIdx%self.numCntrsPerBkt)           

    def incCntrBy1GetVal (self, 
                          cntrIdx  = 0, # idx of the concrete counter to increment in the array 
                          ) -> int: # If forceInc==True, increment the counter. Else, inc the counter w.p. p 
        """
        Increment the counter to the closest higher value.        
        Output:
        cntrDict: a dictionary representing the modified counter where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.
        """
        return self.buckets[self.idx2BucketNum(cntrIdx)].incCntrBy1GetVal (cntrIdx=cntrIdx%self.numCntrsPerBkt)
 

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
        return self.buckets[self.idx2BucketNum(cntrIdx)].incCntrBy1 (cntrIdx=cntrIdx%self.numCntrsPerBkt) 

    def setLogFile (self, logFile):
        """
        set the log file
        """ 
        self.logFile = logFile
        for bkt in self.buckets:
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
            settings.error ('Sorry, Buckets.incCntrGetVal() is currently implemented only when mult==True and factor=1.')
        return self.buckets[self.idx2BucketNum(cntrIdx)].incCntrBy1GetVal (cntrIdx=cntrIdx%self.numCntrsPerBkt) 

