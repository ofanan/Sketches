"""
Parses a .pcap file. 
Output: a csv file, where:
        - the first col. is the keys,
        - the 2nd col. is the id of the clients of this req,
        - the rest of the cols. are the locations ("k_loc") to which a central controller would enter this req. upon a miss. 
"""
from scapy.all import *
import numpy as np
# from numpy import infty
import pandas as pd
import datetime as dt

import settings

def parse_pcap_file (traceFileName     = 'equinix-nyc.dirB.20181220-140100.UTC.anon.pcap',      
                      maxNumOfPkts       = 3, #MyConfig.INF_INT, # maximum number of pkts to be parsed, starting from the beginning of the trace
                      max_trace_len_in_sec = settings.INF_INT, # maximum time length to be parsed, starting from the beginning of the trace 
                      ):

    relativePathToInputFile = settings.getTracesPath() + 'Caida/' + traceFileName       
    settings.checkIfInputFileExists (relativePathToInputFile)    

    for packet in PcapReader(relativePathToInputFile):
        try:
            print(packet[IPv6].src)
        except:
            pass
        
    # shark_cap = rdpcap(relativePathToInputFile)
    # print ('1')
    # pktNum = 0
    #
    # for pkt in shark_cap:
    #     print ('2')
    #     pktNum += 1
    #     if pktNum >= maxNumOfPkts:
    #         break
    #     print (packet.ipv4.src)
    #     # print packet[IPv6].src

parse_pcap_file ()
