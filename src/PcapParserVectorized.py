from scapy.all import *
import numpy as np, mmh3, csv, time
from ttictoc import tic,toc
import pandas as pd
import settings
from settings import *

"""
Parses a .pcap file. 
Output: a csv file, where:
        - the first col. is the keys,
        - the 2nd col. is the id of the clients of this req,
        - the rest of the cols. are the locations ("k_loc") to which a central controller would enter this req. upon a miss. 
"""

# Vectorized function to apply mmh3.hash to each element in the array of the 4-tupes read from the .pcap file.
vectorizedHash = np.vectorize(lambda t: mmh3.hash(t, signed=True), otypes=[np.int32])

def parsePcapFileVectorized (
        traceFileName     = 'equinix-nyc.dirB.20181220-140100.UTC.anon.pcap',      
        maxNumOfPkts     = INF_INT, # maximum number of pkts to be parsed, starting from the beginning of the trace
        maxTraceLenInSec = INF_INT, # maximum time length to be parsed, starting from the beginning of the trace 
    ):
    """
    A vectorized version.
    Unfortunately, this version uses too much run-time memory, and therefore may get stack.
    Parse a .pcap file. Write the parsed file to a .csv file. bearing the same fileName as the .pcap file, but with extension .csv instead of .pcap. 
    """

    tracePath = getTracesPath() + 'Caida' 
    relativePathToInputFile = f'{tracePath}/{traceFileName}'       
    checkIfInputFileExists (relativePathToInputFile)    

    tic ()
    packets = rdpcap(relativePathToInputFile, count=maxNumOfPkts)

    # Determine the number of tuples
    numPkts = len(packets)
    
    # Initialize a NumPy array to store the 4-tuple information
    tuples = np.empty((numPkts), dtype=object)

    # Extract the 4-tuple (source port, dest port, source IP, dest IP) for each packet
    pktNum = 0

    for pkt in packets:
        if pkt.haslayer('IP'):
            if pkt.haslayer('TCP'):
                tuples[pktNum] = ('{}-{}-{}-{}' .format (str(pkt['TCP'].sport), str(pkt['TCP'].dport), str(pkt['IP'].src), str(pkt['IP'].dst)))
                pktNum += 1
            elif pkt.haslayer('UDP'):
                tuples[pktNum] = ('{}-{}-{}-{}' .format (pkt['UDP'].sport, pkt['UDP'].dport, pkt['IP'].src, pkt['IP'].dst))
                pktNum += 1
        if pktNum>=numPkts:
            break

    # Apply the vectorized hash function to the tuple array
    hashArray = vectorizedHash(tuples[:pktNum])

    np.savetxt(f'{tracePath}/{traceFileName}.txt', hashArray, fmt='%d')
    print (f'Finished parsing {numPkts} packets. {genElapsedTimeStr (toc())}')

# parsePcapFile (traceFileName='Caida1_equinix-nyc.dirA.20181220-130000.UTC.anon.pcap', maxNumOfPkts=100000000)
parsePcapFile (traceFileName='Caida2_equinix-chicago.dirA.20160406-130000.UTC.anon.pcap', maxNumOfPkts=100000000)
