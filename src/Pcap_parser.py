"""
Parses a .pcap file. 
Output: a csv file, where:
        - the first col. is the keys,
        - the 2nd col. is the id of the clients of this req,
        - the rest of the cols. are the locations ("k_loc") to which a central controller would enter this req. upon a miss. 
"""
from scapy.all import *
import numpy as np, mmh3, csv, time
# import datetime as dt
# from numpy import infty
import pandas as pd
import settings

def parse_pcap_file (traceFileName     = 'equinix-nyc.dirB.20181220-140100.UTC.anon.pcap',      
                      maxNumOfPkts     = settings.INF_INT, # maximum number of pkts to be parsed, starting from the beginning of the trace
                      maxTraceLenInSec = settings.INF_INT, # maximum time length to be parsed, starting from the beginning of the trace 
                      ):
    """
    Parse a .pcap file. Write the parsed file to a .csv file. bearing the same fileName as the .pcap file, but with extension .csv instead of .pcap. 
    """

    tracePath = settings.getTracesPath() + 'Caida' 
    relativePathToInputFile = f'{tracePath}/{traceFileName}'       
    settings.checkIfInputFileExists (relativePathToInputFile)    

    csvOutputFile   = open(f'{tracePath}/{traceFileName}.csv', 'w', newline='')
    writer          = csv.writer (csvOutputFile)
    pktNum          = 0

    startTime = time.time()            
    for pkt in PcapReader(relativePathToInputFile):
        if IP not in pkt:
            continue
        if TCP in pkt:
            writer.writerow ([mmh3.hash (pkt[IP].src + pkt[IP].dst + str(pkt[TCP].sport) + str(pkt[TCP].sport) + '0') % settings.MAX_NUM_OF_FLOWS]) #([mmh3.hash (pkt[IP].src + pkt[IP].dst + str(pkt[TCP].sport) + str(pkt[TCP].sport) + '0') % settings.MAX_NUM_OF_FLOWS])
        elif UDP in pkt:
            writer.writerow ([mmh3.hash (pkt[IP].src + pkt[IP].dst + str(pkt[UDP].sport) + str(pkt[UDP].sport) + '1') % settings.MAX_NUM_OF_FLOWS])
        else:
            continue
        pktNum += 1    
        if pktNum >= maxNumOfPkts:
            break

    print (f'finished parsing {maxNumOfPkts} pkts by multirow after {time.time() - startTime} sec')


parse_pcap_file ()
