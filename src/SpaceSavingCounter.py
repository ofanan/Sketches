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
    Efficient `Counter`-like structure for approximating the top `m` elements of a stream, in O(m)
    space (https://www.cse.ust.hk/~raywong/comp5331/References/EfficientComputationOfFrequentAndTop-kElementsInDataStreams.pdf).

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

    def _update_element(self, x):
        self._elements_seen += 1

        if x in self._flowSizes:
            self._flowSizes[x] += 1
        elif len(self._flowSizes) < self._cacheSize:
            self._flowSizes[x] = 1
            self._heappush(1, self._elements_seen, x)
        else:
            self._replace_least_element(x)
        print (x, self._flowSizes)

    def _replace_least_element(self, e):
        while True:
            count, tstamp, key = self._heappop()
            assert self._flowSizes[key] >= count

            if self._flowSizes[key] == count:
                break
            else:
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
    

test_SpaceSavingCounter ()    