"""
Parses a .pcap file. 
Output: a csv file, where:
        - the first col. is the keys,
        - the 2nd col. is the id of the clients of this req,
        - the rest of the cols. are the locations ("k_loc") to which a central controller would enter this req. upon a miss. 
"""
from scapy.all import *
import numpy as np, mmh3
# from numpy import infty
# import pandas as pd
import datetime as dt

import settings

def parse_pcap_file (traceFileName     = 'equinix-nyc.dirB.20181220-140100.UTC.anon.pcap',      
                      maxNumOfPkts     = 300000, #MyConfig.INF_INT, # maximum number of pkts to be parsed, starting from the beginning of the trace
                      maxTraceLenInSec = settings.INF_INT, # maximum time length to be parsed, starting from the beginning of the trace 
                      ):

    relativePathToInputFile = settings.getTracesPath() + 'Caida/' + traceFileName       
    settings.checkIfInputFileExists (relativePathToInputFile)    

    pktNum = 0
    for pkt in PcapReader(relativePathToInputFile):
        pktNum += 1
        if pktNum >= maxNumOfPkts:
            break
        if IP not in pkt:
            continue
        if TCP in pkt:
            key = mmh3.hash (pkt[IP].src + pkt[IP].dst + str(pkt[TCP].sport) + str(pkt[TCP].sport) + '0') % settings.MAX_NUM_OF_FLOWS
        elif UDP in pkt:
            key = mmh3.hash (pkt[IP].src + pkt[IP].dst + str(pkt[UDP].sport) + str(pkt[UDP].sport) + '1') % settings.MAX_NUM_OF_FLOWS
        else:
            continue
        print (key)
        # print (pkt.summary())
        # if TCP in pkt:
        #     print ('TCP')
        # elif UDP in pkt:
        #     print ('UDP')
        # else:
        #     settings.error ('else')
        # try:
        #     # flowId = mmh3.hash (pkt[IP].src)
        #     print(pkt[UDP].dport)
        #     # print (flowId)
        #     # print(pkt[TCP].dport)
        #     # print(pkt[IP].src)
        #     # print(pkt[IP].dst)
        # except:
        #     pass
        
    # shark_cap = rdpcap(relativePathToInputFile)
    # print ('1')
    #
    # for pkt in shark_cap:
    #     print ('2')
    #     print (packet.ipv4.src)
    #     # print packet[IPv6].src

parse_pcap_file ()
