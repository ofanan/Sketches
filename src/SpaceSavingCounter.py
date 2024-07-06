# https://codereview.stackexchange.com/questions/205090/spacesaving-frequent-item-counter-in-python 

class SpaceSavingCounter:
    """
    Efficient `Counter`-like structure for approximating the top `m` elements of a stream, in O(m)
    space (https://www.cse.ust.hk/~raywong/comp5331/References/EfficientComputationOfFrequentAndTop-kElementsInDataStreams.pdf).

    Specifically, the resulting counter will contain the correct counts for the top k elements with
    k ≈ m.  The interface is the same as `collections.Counter`.
    """

    def __init__(self, m):
        self._m = m
        self._elements_seen = 0
        self._counts = Counter()  # contains the counts for all elements
        self._queue = []  # contains the estimated hits for the counted elements

    def _update_element(self, x):
        self._elements_seen += 1

        if x in self._counts:
            self._counts[x] += 1
        elif len(self._counts) < self._m:
            self._counts[x] = 1
            self._heappush(1, self._elements_seen, x)
        else:
            self._replace_least_element(x)

    def _replace_least_element(self, e):
        while True:
            count, tstamp, key = self._heappop()
            assert self._counts[key] >= count

            if self._counts[key] == count:
                break
            else:
                self._heappush(self._counts[key], tstamp, key)

        del self._counts[key]
        self._counts[e] = count + 1
        self._heappush(count, self._elements_seen, e)

    def _heappush(self, count, tstamp, key):
        heapq.heappush(self._queue, (count, tstamp, key))

    def _heappop(self):
        return heapq.heappop(self._queue)
    
    
        def most_common(self, n=None):
        return self._counts.most_common(n)

    def elements(self):
        return self._counts.elements()

    def __len__(self):
        return len(self._counts)

    def __getitem__(self, key):
        return self._counts[key]

    def __iter__(self):
        return iter(self._counts)

    def __contains__(self, item):
        return item in self._counts

    def __reversed__(self):
        return reversed(self._counts)

    def items(self):
        return self._counts.items()

    def keys(self):
        return self._counts.keys()

    def values(self):
        return self._counts.values()
    
    
    def test_SpaceSavingCounter():
    ssc = SpaceSavingCounter(2)
    ssc.update([1, 5, 3, 4, 2, 7, 7, 1, 3, 1, 3, 1, 3, 1, 3])
    assert ssc.keys() == {1, 3}

    ssc = SpaceSavingCounter(2)
    ssc.update([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2])
    assert ssc.keys() == {3, 2}

    ssc = SpaceSavingCounter(1)
    ssc.update([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2])
    assert ssc.keys() == {2}

    ssc = SpaceSavingCounter(3)
    ssc.update([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2])
    assert ssc.keys() == {1, 2, 3}

    ssc = SpaceSavingCounter(2)
    ssc.update([])
    assert ssc.keys() == set()
    
    