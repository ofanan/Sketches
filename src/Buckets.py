# Buckets of Counter arrays.
# Each bucket is composed of several counters, and some variable/parameter, shared by all the counters in that bucket.
# The shared variable may be the exponent value (in the buckets are SEC buckets), the Epsilon step (if these are ICE buckets), etc.
import matplotlib 
import matplotlib.pyplot as plt
import math, random, os, pickle, mmh3, time
import numpy as np
from datetime import datetime

import settings, SEC, IceBucket, MecBucket 
# import F2pBucket #$$ currently private
from printf import printf, printarFp

class Buckets (object):
    """
    Buckets of counters, sharing the scaling/"epsilon" parameter.
    """

    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr  = lambda self : f'bkts{self.numBkts}'

    # Given the index in the Buckets, get the bucket number and the number of the counter in the bucket
    getBucketNumAndCntrNum = lambda self, idx : [self.idx2BucketNum(idx), idx%self.numCntrsPerBkt]
       
    # Given the index in the Buckets, get the bucket number 
    idx2BucketNum = lambda self, idx : idx//self.numCntrsPerBkt
    
    # Query a cntr. 
    # Input: cntrIdx - the counter's index. 
    # Output: The value that the counter represents (as int/FP).
    queryCntrVal    = lambda self, cntrIdx=0 : self.bkts[self.idx2BucketNum(cntrIdx)].queryCntrVal(cntrIdx=cntrIdx%self.numCntrsPerBkt)                  
            
    def __init__ (self, 
                  cntrSize          = 4, # num of bits in each counter. 
                  numCntrs          = 9, # number of counters in the array.
                  cntrMaxVal        = None, 
                  numCntrsPerBkt    = 1, # number of cntrs at each bucket.
                  mode              = 'SEC',
                  numEpsilonSteps   = 8,
                  verbose           = [], # determines which outputs would be written to .log/.res/.pcl/debug files, as detailed in settings.py.
                  ):

        if cntrSize<1 or numCntrs<1:
            settings.error (f'in Buckets: you requested cntrSize={cntrSize}, numCntrs={numCntrs}. However, you should choose cntrSize>=1, numCntrs>=1.')
            
        self.cntrSize, self.numCntrs, self.numCntrsPerBkt = int(cntrSize), int(numCntrs), int(numCntrsPerBkt)
        self.numBkts = self.numCntrs // self.numCntrsPerBkt
        self.verbose    = verbose
        self.mode       = mode
        if mode=='SEC':
            self.bkts = [SEC.CntrMaster(cntrSize         = self.cntrSize, 
                                numCntrs         = self.numCntrsPerBkt, 
                                verbose          = self.verbose) 
                            for _ in range (self.numBkts)]
        elif mode=='ICE':
            self.bkts = [IceBucket.CntrMaster(
                                cntrSize        = self.cntrSize, 
                                numCntrs        = self.numCntrsPerBkt,
                                cntrMaxVal      = cntrMaxVal, 
                                numEpsilonSteps = numEpsilonSteps,
                                verbose         = self.verbose,
                                id              = i) 
                            for i in range (self.numBkts)]
        elif mode=='F2P':
            settings.error ('Sorry, F2P buckets are private.')
            # self.bkts = [F2pBucket.CntrMaster(
            #                     cntrSize        = self.cntrSize, 
            #                     numCntrs        = self.numCntrsPerBkt,
            #                     hyperExpSize    = 0,
            #                     verbose         = self.verbose) 
            #                 for _ in range (self.numBkts)]
        elif mode=='MEC':
            numStages = int(25)
            MecBucket.CntrMaster.expRanges, MecBucket.CntrMaster.offsets, MecBucket.CntrMaster.pivots = \
                MecBucket.precomputeExpRangesAndOffsets (cntrSize=self.cntrSize, numStages=numStages)
            MecBucket.CntrMaster.numStages = numStages
            self.bkts = [MecBucket.CntrMaster(
                                            cntrSize        = self.cntrSize, 
                                            numCntrs        = self.numCntrsPerBkt,
                                            verbose=self.verbose) for _ in range (self.numBkts)]
        else:
            settings.error ('Sorry. Mode {self.mode} which you chose is not supported yet by Buckets.py.')
        
        
    def printCntrsStat (self, 
                        outputFile, # file to which the stat will be written
                        genPlot=False, # when True, plot the stat 
                        outputFileName=None, # filename to which the .pdf plot will be saved
                        ) -> None:
        """
        Print statistics about the counters, e.g., the max counter, and binning of the counters.
        """
        if self.mode!='ICE':
            print ('Sorry. Printing stat is currently implemented only for IceBuckets.')
            return
        cntrVals = [None]*self.numCntrs
        i = 0
        for bktNum in range(self.numBkts):            
            cntrVals[i:(i+self.numCntrsPerBkt)] = self.bkts[bktNum].getAllCntrsVals()
            i += self.numCntrsPerBkt
        maxCntr = max(cntrVals)
        printf (outputFile, f'numBkts={self.numBkts}, numCntrs={self.numCntrs}, maxCntr={maxCntr}\n')

        numBins = min (100, maxCntr+1)
        binSize = maxCntr // (numBins-1)
        binVal  = [None] * numBins 
        for bin in range(numBins):
            binVal[bin] = len ([cntrNum for cntrNum in range(self.numCntrs) if (cntrVals[cntrNum]//binSize)==bin])
        binFlowSizes = [binSize*bin for bin in range (numBins)]
        printf (outputFile, f'binVal={binVal}')
        printf (outputFile, f'\nbinFlowSizes={binFlowSizes}')
        printf (outputFile, f'\ncntrVals={cntrVals}\n')
        if not(genPlot):
            return 
        if outputFileName==None:
            settings.error (f'In Buckets.printCntrsStat(). To generate a plot, please specify outputFileName')
        _, ax = plt.subplots()
        ax.plot ([binSize*bin for bin in range (numBins)], binVal)
        ax.set_yscale ('log')
        plt.savefig (f'../res/{outputFileName}.pdf', bbox_inches='tight')        
        
    
    def printAllCntrs (self, outputFile) -> None:
        """
        Format-print all the counters as a single the array, to the given file.
        """
        for bkt in self.bkts:
            bkt.printAllCntrVals(outputFile)
    
    def rstCntr (self, cntrIdx=0) -> None:
        """
        Reset a single counter.
        """
        self.bkts[self.idx2BucketNum(cntrIdx)].rstCntr(idx%self.numCntrsPerBkt)
    
    def queryCntr (self, cntrIdx=0) -> dict:
        """
        Query a cntr.
        Input: 
        cntrIdx - the counter's index. 
        Output:
        cntrDic: a dictionary, where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.        
        """
        return self.bkts[self.idx2BucketNum(cntrIdx)].cntr2cntrDict(cntrIdx%self.numCntrsPerBkt)           

    def incCntrBy1GetVal (self, 
                          cntrIdx  = 0, # idx of the concrete counter to increment in the array 
                          ) -> int: # If forceInc==True, increment the counter. Else, inc the counter w.p. p 
        """
        Increment the counter to the closest higher value.        
        Output:
        cntrDict: a dictionary representing the modified counter where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.
        """
        return self.bkts[self.idx2BucketNum(cntrIdx)].incCntrBy1GetVal (cntrIdx=cntrIdx%self.numCntrsPerBkt)
 

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
        return self.bkts[self.idx2BucketNum(cntrIdx)].incCntrBy1 (cntrIdx=cntrIdx%self.numCntrsPerBkt) 

    def setLogFile (self, logFile):
        """
        set the log file
        """ 
        self.logFile = logFile
        for bkt in self.bkts:
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
        return self.bkts[self.idx2BucketNum(cntrIdx)].incCntrBy1GetVal (cntrIdx=cntrIdx%self.numCntrsPerBkt) 

