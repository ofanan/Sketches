import heapq
from collections import defaultdict

class SpaceSaving:
    def __init__(self, cacheSize):
        self.cacheSize = cacheSize
        self.CntrsAr = defaultdict(int)
        self.minHeap = []

    def process(self, item):
        if item in self.CntrsAr:
            self.CntrsAr[item] += 1
        elif len(self.minHeap) < self.cacheSize:
            self.CntrsAr[item] = 1
            heapq.heappush(self.minHeap, (1, item))
        else:
            min_count, min_item = heapq.heappop(self.minHeap)
            del self.CntrsAr[min_item]
            self.CntrsAr[item] = min_count + 1
            heapq.heappush(self.minHeap, (min_count + 1, item))
            self.printHeap()

    def printHeap (self):
        for item in self.minHeap:
            print (f'{item} ')
        print ('')

    def get_top_k(self):
        return sorted(self.CntrsAr.items(), key=lambda x: -x[1])

# Usage example
if __name__ == "__main__":
    cacheSize = 5  # Top k elements to keep track of
    stream = ['a', 'b', 'c', 'a', 'b', 'a', 'd', 'e', 'e', 'e', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'a', 'b', 'c']

    ss = SpaceSaving(cacheSize)
    for item in stream:
        ss.process(item)

    top_k = ss.get_top_k()
    print("Top-k elements:", top_k)
