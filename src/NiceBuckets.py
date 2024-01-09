# This class implements NiceBuckets (New Ice Buckets).
# ICE_buckets are detailed in the paper: "Independent counter estimation buckets", Einziger, Gil and Fellman, Benny and Kassner, Yaron, Infocom'12.
# A nice bucket is an IceBucket with improved capabilities (to be added later), e.g., tuning determination of Epsilon in a more efficient way.
# The array of Ice Buckets consists of Reg (regular) NiceBuckets to count mice, plus Xl Nicebuckets, to count elephants. 
import matplotlib 
import matplotlib.pyplot as plt
import math, random, os, pickle, mmh3, time
import numpy as np
from datetime import datetime

import settings, Buckets, IceBucket, NiceBucket
from printf import printf, printarFp

class CntrMaster (Buckets.Buckets):
    """
    New (improved) IceBuckets.
    """

    genSettingsStr = lambda self : 'Nice_n{}'.format(self.cntrSize)

    # Given the index in the Buckets, get the regular bucket number 
    idx2RegBktNum = lambda self, idx : idx//self.numCntrsPerRegBkt

    # Given the index in the Buckets, get the XL bucket number 
    idx2XlBktNum = lambda self, idx : idx//self.numIndicesPerXlBkt

    def queryCntrVal (self, cntrIdx=0):
        """
        Query a cntr. 
        Input: cntrIdx - the counter's index. 
        Output: The value that the counter represents (as int/FP).
        """
        val = self.regBkts[self.idx2RegBktNum(cntrIdx)].queryCntrVal(cntrIdx=cntrIdx%self.numCntrsPerRegBkt)
        if val==self.minValOfXlBkt:
            return self.xlBkts[self.idx2XlBktNum (cntrIdx)].queryCntrVal (cntrIdx=cntrIdx%self.numCntrsPerXlBkt) + self.minValOfXlBkt
        return val
            
    def __init__ (self, 
                  cntrSize                  = 4, # num of bits in each counter. 
                  numCntrs                  = 9, # number of counters in the array.
                  numCntrsPerRegBkt         = 1, # number of cntrs at each bucket.
                  numCntrsPerXlBkt          = 1,
                  numXlBkts                 = 1,
                  numEpsilonStepsInRegBkt   = 4,
                  numEpsilonStepsInXlBkt    = 4,
                  verbose                   = [], # determines which outputs would be written to .log/.res/.pcl/debug files, as detailed in settings.py.
                  ):

        if cntrSize<1 or numCntrs<1:
            settings.error (f'in Buckets: you requested cntrSize={cntrSize}, numCntrs={numCntrs}. However, you should choose cntrSize>=1, numCntrs>=1.')
            
        self.cntrSize, self.numCntrs, self.numCntrsPerRegBkt = int(cntrSize), int(numCntrs), int(numCntrsPerRegBkt)
        self.numEpsilonStepsInRegBkt = numEpsilonStepsInRegBkt
        self.numCntrsPerXlBkt = numCntrsPerXlBkt
        self.numRegBkts = self.numCntrs // self.numCntrsPerRegBkt
        self.verbose    = verbose
        self.numXlBkts  = numXlBkts
        self.mode       = 'Nice'
        self.numIndicesPerXlBkt = int (math.ceil(self.numCntrs / self.numXlBkts)) 
        self.minValOfXlBkt = IceBucket.calcCntrMaxValsByCntrSizes (numEpsilonSteps=self.numEpsilonStepsInRegBkt, cntrSize=self.cntrSize)[self.numEpsilonStepsInRegBkt-1] 
        self.regBkts = [NiceBucket.CntrMaster(
                            cntrSize        = self.cntrSize, 
                            numCntrs        = self.numCntrsPerRegBkt,
                            numEpsilonSteps = self.numEpsilonStepsInRegBkt,
                            verbose         = self.verbose,
                            id              = i,
                            isXlBkt         = False)
                            for i in range (self.numRegBkts)]        
        self.xlBkts = [NiceBucket.CntrMaster(
                            cntrSize        = self.cntrSize, 
                            numCntrs        = self.numCntrsPerXlBkt,
                            numEpsilonSteps = numEpsilonStepsInXlBkt,
                            verbose         = self.verbose,
                            id              = i,
                            isXlBkt         = True)
                            for i in range (self.numXlBkts)]
        
    def printAllCntrs (self, outputFile) -> None:
        """
        Format-print all the counters as a single the array, to the given file.
        """
        printf (outputFile, 'Reg bkts:\n')
        for bkt in self.regBkts:
            bkt.printAllCntrVals(outputFile)
        printf (outputFile, 'Xl bkts:\n')
        for bkt in self.xlBkts:
            bkt.printAllCntrVals(outputFile)
    
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
        val = self.regBkts[self.idx2RegBktNum(cntrIdx)].cntr2cntrDict(cntrIdx%self.numCntrsPerRegBkt)
        if val==self.minValOfXlBkt:
            settings.error ('reached the max val of regular bkts')
        return val

    def incCntrBy1GetVal (self, 
                          cntrIdx  = 0, # idx of the concrete counter to increment in the array 
                          ) -> int: # If forceInc==True, increment the counter. Else, inc the counter w.p. p 
        """
        Increment the counter to the closest higher value.        
        Output:
        cntrDict: a dictionary representing the modified counter where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.
        """
        regBktNum = self.idx2RegBktNum(cntrIdx)
        isSaturated, valAfterInc = self.regBkts[self.idx2RegBktNum(cntrIdx)].incCntrBy1GetVal (cntrIdx=cntrIdx%self.numCntrsPerRegBkt)
        if isSaturated:
            # Regular value is saturated --> query the Xl bkt
            _, valAfterInc = self.xlBkts[self.idx2XlBktNum (cntrIdx)].incCntrBy1GetVal (cntrIdx=cntrIdx%self.numCntrsPerXlBkt) 
            return valAfterInc + self.minValOfXlBkt
        return valAfterInc

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
        settings.error ('Sorry, Buckets.incCntrBy1 () is not implemented yet.')

    def setLogFile (self, logFile):
        """
        set the log file
        """ 
        self.logFile = logFile
        for bkt in self.regBkts:
            bkt.logFile = logFile
        for bkt in self.xlBkts:
            bkt.logFile = logFile

    def printCntrsStat (self, 
                        outputFile, # file to which the stat will be written
                        genPlot=False, # when True, plot the stat 
                        outputFileName=None, # filename to which the .pdf plot will be saved
                        ) -> None:
        """
        Print statistics about the counters, e.g., the max counter, and binning of the counters.
        """
        cntrVals = [None]*self.numCntrs
        i = 0
        for bktNum in range(self.numRegBkts):            
            cntrVals[i:(i+self.numCntrsPerRegBkt)] = self.regBkts[bktNum].getAllCntrsVals()
            i += self.numCntrsPerRegBkt
        maxCntr = max(cntrVals)
        printf (outputFile, f'numRegBkts={self.numRegBkts}, numCntrs={self.numCntrs}, maxCntr={maxCntr}\n')

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
