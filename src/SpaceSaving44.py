import heapq
from collections import defaultdict

class SpaceSaving:
    def __init__(self, cacheSize):
        self.cacheSize = cacheSize
        self.counter = defaultdict(int)
        self.minHeap = []

    def process(self, item):
        if item in self.counter:
            self.counter[item] += 1
        elif len(self.minHeap) < self.cacheSize:
            self.counter[item] = 1
            heapq.heappush(self.minHeap, (1, item))
        else:
            min_count, min_item = heapq.heappop(self.minHeap)
            del self.counter[min_item]
            self.counter[item] = min_count + 1
            heapq.heappush(self.minHeap, (min_count + 1, item))

    def get_top_k(self):
        return sorted(self.counter.items(), key=lambda x: -x[1])

# Usage example
if __name__ == "__main__":
    cacheSize = 3  # Top k elements to keep track of
    stream = ['a', 'b', 'c', 'a', 'b', 'a', 'd', 'e', 'e', 'e', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'a', 'b', 'c']

    ss = SpaceSaving(cacheSize)
    for item in stream:
        ss.process(item)

    top_k = ss.get_top_k()
    print("Top-k elements:", top_k)
