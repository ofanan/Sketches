"""
Parses a .pcap file. 
Output: a csv file, where:
        - the first col. is the keys,
        - the 2nd col. is the id of the clients of this req,
        - the rest of the cols. are the locations ("k_loc") to which a central controller would enter this req. upon a miss. 
"""
from scapy.all import *
import numpy as np, mmh3, csv, time
import pandas as pd
import settings
from settings import *

def parse_pcap_file (
        traceFileName     = 'equinix-nyc.dirB.20181220-140100.UTC.anon.pcap',      
        maxNumOfPkts     = INF_INT, # maximum number of pkts to be parsed, starting from the beginning of the trace
        maxTraceLenInSec = INF_INT, # maximum time length to be parsed, starting from the beginning of the trace 
    ):
    """
    Parse a .pcap file. Write the parsed file to a .csv file. bearing the same fileName as the .pcap file, but with extension .csv instead of .pcap. 
    """

    tracePath = getTracesPath() + 'Caida' 
    relativePathToInputFile = f'{tracePath}/{traceFileName}'       
    checkIfInputFileExists (relativePathToInputFile)    

    csvOutputFile   = open(f'{tracePath}/{traceFileName}.csv', 'w', newline='')
    writer          = csv.writer (csvOutputFile)
    pktNum          = 0

    print (f'Started parsing {traceFileName}')
    startTime = time.time()            
    for pkt in PcapReader(relativePathToInputFile):
        if IP not in pkt:
            continue
        if TCP in pkt:
            mmh3.hash ('{}-{}-{}-{}' .format (
                str(pkt['TCP'].sport), 
                str(pkt['TCP'].dport), 
                str(pkt['IP'].src), 
                str(pkt['IP'].dst))) % MAX_NUM_OF_FLOWS
        elif UDP in pkt:
            mmh3.hash ('{}-{}-{}-{}' .format (
                str(pkt['UDP'].sport), 
                str(pkt['UDP'].dport), 
                str(pkt['IP'].src), 
                str(pkt['IP'].dst))) % MAX_NUM_OF_FLOWS
        else:
            continue
        pktNum += 1    
        if pktNum >= maxNumOfPkts:
            break

    print (f'finished parsing {pktNum} pkts by multirow after {time.time() - startTime} sec')


maxNumOfPkts = 100000000
parse_pcap_file (traceFileName='Caida1_equinix-nyc.dirA.20181220-130000.UTC.anon.pcap', maxNumOfPkts=maxNumOfPkts)
# parse_pcap_file (traceFileName='Caida2_equinix-chicago.dirA.20160406-130000.UTC.anon.pcap', maxNumOfPkts=maxNumOfPkts)