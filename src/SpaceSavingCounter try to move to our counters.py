# https://codereview.stackexchange.com/questions/205090/spacesaving-frequent-item-counter-in-python 
import threading, heapq 
import numpy as np
from threading import Thread
from datetime import datetime
from collections import defaultdict, Counter 

import settings #, PerfectCounter, Buckets, NiceBuckets, 
from settings import *
from   tictoc import tic, toc # my modules for measuring and print-out the simulation time.
from printf import printf, printarFp 
from SingleCntrSimulator import getFxpCntrMaxVal, genCntrMasterFxp
from _ast import Or

class SpaceSavingCounter:
    """
    Efficient `Counter`-like structure for approximating the top `m` elements of a stream, in O(m) space 
    (https://www.cse.ust.hk/~raywong/comp5331/References/EfficientComputationOfFrequentAndTop-kElementsInDataStreams.pdf).

    Specifically, the resulting counter will contain the correct counts for the top k elements with
    k ≈ m.  The interface is the same as `collections.Counter`.
    """

    def __init__(
            self, 
            cacheSize   : int  = 1,
            verbose     : list = [],
        ):
        self._cacheSize     = cacheSize
        self._elements_seen = 0
        self._flowSizes     = Counter()  # contains the counts for all elements
        self._queue         = []  # contains the estimated hits for the counted elements
        self.verbose        = verbose

    def incNQueryFlow(
            self, 
            flowId
        ):
        """
        Update the value for a single flow. Return the updated estimated value for this flow.
        To ease the finding of min item (without the need to perform cntr2num), we cache also the cached values.  
        """
        self._elements_seen += 1
        hit = False
        if flowId in self._flowSizes: # is x cached?
            self._flowSizes[x] += 1
        for cntrIdx in range(self.numCntrs): # loop over the cache's elements
            if self.flowIds[cntrIdx]==flowId: # found the flowId in the $
                self.flowSizes[cntrIdx] = int(round(self.cntrMaster.incCntrBy1GetVal (cntrIdx=cntrIdx))) # prob-inc. the counter, and get its val
                hit = True 
                break
            elif self.flowIds[cntrIdx]==None: # the flowId isn't cached yet, and the $ is not full yet
                self.flowIds  [cntrIdx] = flowId # insert flowId into the $
                self.flowSizes[cntrIdx] = int(round(self.cntrMaster.incCntrBy1GetVal (cntrIdx=cntrIdx))) # prob-inc. the counter, and get its val
                hit = True 
                break
        if not(hit): # didn't find flowId in the $ --> insert it
            cntrIdx = min(range(self.numCntrs), key=self.flowSizes.__getitem__) # find the index of the minimal cached item # to allow randomizing between all minimal items, np.where(a==a.min())
            self.flowIds  [cntrIdx] = flowId # replace the item by the newly-inserted flowId
            self.flowSizes[cntrIdx] = int(round(self.cntrMaster.incCntrBy1GetVal (cntrIdx=cntrIdx))) # prob'-inc. the value
        return self.flowSizes[cntrIdx]

    def _update_element(self, x):

        if x in self._flowSizes: # is x cached?
            self._flowSizes[x] += 1
        elif len(self._flowSizes) < self._cacheSize: # x isn't in the $, but the $ is not full --> insert x
            self._flowSizes[x] = 1
            self._heappush(1, self._elements_seen, x)
        else:
            self._replace_least_element(x)
        # print (x, self._flowSizes)

    def _replace_least_element(self, e):
        while True:
            count, tstamp, key = self._heappop() # count should be the lowest count of a key in the heap
            assert self._flowSizes[key] >= count # Verify that indeed the $ed value of this item is >= count.

            if self._flowSizes[key] == count: # the val of this key wasn't incremented since its last push --> this is surely the least element.
                break
            else: # The real val of this key was incremented since it was cached --> re-push it to the correct place, with the value it should had.
                self._heappush(self._flowSizes[key], tstamp, key)

        del self._flowSizes[key]
        self._flowSizes[e] = count + 1
        self._heappush(count, self._elements_seen, e)

    def _heappush(self, count, tstamp, key):
        heapq.heappush(self._queue, (count, tstamp, key))

    def _heappop(self):
        return heapq.heappop(self._queue)
    
    
    def most_common(self, n=None):
        return self._flowSizes.most_common(n)

    def elements(self):
        return self._flowSizes.elements()

    def __len__(self):
        return len(self._flowSizes)

    def __getitem__(self, key):
        return self._flowSizes[key]

    def __iter__(self):
        return iter(self._flowSizes)

    def __contains__(self, item):
        return item in self._flowSizes

    def __reversed__(self):
        return reversed(self._flowSizes)

    def items(self):
        return self._flowSizes.items()

    def keys(self):
        return self._flowSizes.keys()

    def values(self):
        return self._flowSizes.values()
    
    def printSimMsg (self, str):
        """
        Print-screen an info msg about the parameters and hours of the simulation starting to run. 
        """             
        print ('{} running ss at t={}. trace={}, numOfExps={}, mode={}, cntrSize={}, cacheSize={}' .format (
                        str, datetime.now().strftime('%H:%M:%S'), self.traceFileName, self.numOfExps, self.mode, self.cntrSize, self.numCntrs))


    def fillStatDictsFields (self, dict) -> dict:
        """
        Add to the given dict some fields detailing the sim settings. Return the full dict.
        """
        dict['numOfExps']   = self.expNum+1# The count of the experiments started in 0
        dict['numIncs']     = self.incNum
        dict['mode']        = self.mode
        dict['cacheSize']   = self.numCntrs
        dict['numFlows']    = self.numFlows
        dict['cntrSize']    = self.cntrSize
        dict['seed']        = self.seed
        dict['maxValBy']    = self.maxValBy
        return dict
    
    def sim (
        self, 
        traceName   = None,
        ):
        """
        Simulate the count min sketch
        """
        
        self.sumSqAbsEr  = np.zeros (self.numOfExps) # self.sumSqAbsEr[j] will hold the sum of the square absolute errors collected at experiment j. 
        self.sumSqRelEr  = np.zeros (self.numOfExps) # self.sumSqRelEr[j] will hold the sum of the square relative errors collected at experiment j.        
        self.printSimMsg ('Started')
        self.openOutputFiles ()
        tic ()
        if self.numFlows==None:
            error ('In CountMinSketch.runSimFromTrace(). Sorry, dynamically calculating the flowNum is not supported yet.')

        if self.traceName=='Rand': # run synthetic, randomized trace
            rng = np.random.default_rng()
            self.trace = rng.integers (self.numFlows, size=self.maxNumIncs, dtype='uint32')
        else:
            relativePathToInputFile = getRelativePathToTraceFile (f'{getTraceFullName(self.traceName)}.txt')
            checkIfInputFileExists (relativePathToInputFile, exitError=True)
            self.trace = np.fromfile(relativePathToInputFile, count = self.maxNumIncs, sep='\n', dtype='uint32')
        self.maxNumIncs = min (self.maxNumIncs, self.trace.shape[0]) 

        for self.expNum in range (self.numOfExps):
            self.seed = self.expNum+1
            random.seed (self.seed) 
            self.genCntrMaster () # Generate a fresh, empty CntrMaster, for each experiment
            self.cntrMaster.setLogFile(self.logFile)
            flowRealVal = np.zeros(self.numFlows)
            self.writeProgress () # log the beginning of the experiment; used to track the progress of long runs.

            for self.incNum in range(self.maxNumIncs):
                flowId = self.trace[self.incNum]            
                flowRealVal[flowId]     += 1
                flowEstimatedVal = self.incNQueryFlow (flowId)
                sqEr = (flowRealVal[flowId] - flowEstimatedVal)**2
                self.sumSqAbsEr[self.expNum] += sqEr    
                self.sumSqRelEr[self.expNum] += sqEr/(flowRealVal[flowId])**2                
                if VERBOSE_LOG_SHORT in self.verbose: 
                    self.cntrMaster.printAllCntrs (self.logFile, printAlsoVec=False)
                    printf (self.logFile, 'incNum={}, hashes={}, estimatedVal={:.0f} realVal={:.0f} \n' .format(
                        self.incNum, traceHashes[self.incNum], flowEstimatedVal, flowRealVal[flowId]))
                elif VERBOSE_LOG in self.verbose: 
                    self.cntrMaster.printAllCntrs (self.logFile, printAlsoVec=True)
                    printf (self.logFile, 'incNum={}, hashes={}, estimatedVal={:.0f} realVal={:.0f} \n' .format(
                        self.incNum, traceHashes[self.incNum], flowEstimatedVal, flowRealVal[flowId]))
                if VERBOSE_DETAILED_LOG in self.verbose and self.incNum>10000: #$$$
                    printf (self.logFile, 'incNum={}, realVal={}, estimated={:.1e}, sqAbsEr={:.1e}, sqRelEr={:.1e}, sumSqAbsEr={:.1e}, sumSqRelEr={:.1e}\n' .format (
                        self.incNum, flowRealVal[flowId], flowEstimatedVal, sqAbsEr, sqRelEr, self.sumSqAbsEr[self.expNum], self.sumSqRelEr[self.expNum]))
            if self.expNum==0: # Log (if at all) only in the first experiment. No need to log again in further exps.
                self.logEndSim ()
                self.rmvVerboseLogs ()
        for rel_abs_n in [True, False]:
            for statType in ['Mse', 'normRmse']:
                sumSqEr = self.sumSqRelEr if rel_abs_n else self.sumSqAbsEr
                dict = calcPostSimStat(
                    sumSqEr         = sumSqEr, 
                    statType        = statType, 
                    numMeausures    = self.incNum+1,
                    verbose         = self.verbose,
                    logFile         = self.logFile
                    )
                dict = self.fillStatDictsFields(dict)
                dict['rel_abs_n']   = rel_abs_n
                
                if VERBOSE_PCL in self.verbose:
                    self.dumpDictToPcl    (dict)
                if VERBOSE_RES in self.verbose:
                    printf (self.resFile, f'{dict}\n\n') 
        print (f'Finished {self.incNum+1} increments. {genElapsedTimeStr (toc())}')

def test_SpaceSavingCounter():
    ssc = SpaceSavingCounter(3)
    # testTrace = [1, 5, 3, 4, 2, 7, 7, 1, 3, 1, 3, 1, 3, 1, 3]
    testTrace = [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2]
    # ssc.update()
    for flowId in testTrace:
        ssc._update_element(flowId) 

    # ssc = SpaceSavingCounter(2)
    # assert ssc.keys() == {3, 2}
    #
    # ssc = SpaceSavingCounter(1)
    # ssc.update([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2])
    # assert ssc.keys() == {2}
    #
    # ssc = SpaceSavingCounter(3)
    # ssc.update([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2])
    # assert ssc.keys() == {1, 2, 3}
    #
    # ssc = SpaceSavingCounter(2)
    # ssc.update([])
    # assert ssc.keys() == set()
# test_SpaceSavingCounter ()

def LaunchSsSim (
        traceFileName   : str, 
        cntrSize        : int, 
        mode            : str, # a string, detailing the mode of the counter, e.g. "F2P_li_h2".
        maxNumIncs      : float = float ('inf'), 
        cacheSize       : int,
    ):
    """
    Lanuch a simulation of Space Saving.
    """
    if traceFileName=='Rand':
        ss = SpaceSaving (
            numFlows        = 9,
            cntrSize        = cntrSize, 
            cacheSize       = 3,
            verbose         = [VERBOSE_LOG, VERBOSE_LOG_DWN_SMPL], # VERBOSE_LOG, VERBOSE_LOG_END_SIM, VERBOSE_LOG, VERBOSE_DETAILS
            traceFileName   = traceFileName,
            mode            = mode,
            numOfExps       = 1, 
            maxNumIncs      = 33,
            maxValBy        = 'F2P_li_h2',
        )
    else:
        ss = SpaceSaving (
            cntrSize        = cntrSize,
            numFlows        = getNumFlowsByTraceName (traceFileName), 
            cacheSize       = cacheSize,
            verbose         = [VERBOSE_RES, VERBOSE_PCL, VERBOSE_LOG_END_SIM, VERBOSE_LOG_DWN_SMPL], # [VERBOSE_RES, VERBOSE_PCL] # VERBOSE_LOG_END_SIM,  VERBOSE_RES, VERBOSE_FULL_RES, VERBOSE_PCL] # VERBOSE_LOG, VERBOSE_RES, VERBOSE_PCL, VERBOSE_DETAILS
            mode            = mode,
            traceFileName   = traceFileName,
            numOfExps       = 10, 
            maxValBy        = 'F2P_li_h2',
        )
    ss.sim ()


if __name__ == '__main__':
    try:
        for cacheSize in [2**i for i in range(10, 19)]:
            for traceFileName in ['Caida1', 'Caida2']:
                LaunchSsSim (
                    traceFileName   = traceFileName, 
                    cntrSize        = 8, 
                    mode            = 'F2P_li_h2_ds', # a string, detailing the mode of the counter, e.g. "F2P_li_h2".
                    maxNumIncs      = maxNumIncs, 
                    cacheSize       = cacheSize
                )

    except KeyboardInterrupt:
        print('Keyboard interrupt.')
    