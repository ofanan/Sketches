"""
Parses a .pcap file. 
"""
from scapy.all import *
import numpy as np, mmh3, csv, pandas as pd
from ttictoc import tic,toc
import settings
from settings import *

def processPcapAndWriteHashes(
        outputFileName      : str,       
        pcapFileNames       : list = [], # pcap files to parse
        maxNumOfPkts         = INF_INT, # maximum number of pkts to be parsed, starting from the beginning of the trace
        parseNonTcpUdpIpPkts = False, # When True, insert to the tuple also IP pkts that aren't TCP/UDP  
        parseNonIpPkts       = False, # When True, insert to the tuple also non-IP pkts   
    ):
    """
    Parse a .pcap file. Write the parsed file to a .csv file. bearing the same fileName as the .pcap file, but with extension .csv instead of .pcap. 
    """

    tracePath = getTracesPath() + 'Caida' 
    outputFile = open (f'{tracePath}/{outputFileName}', 'a+') 
    
    # Initialize counters
    totalPkts        = 0
    nonTcpUdpIpPkts  = 0
    nonIpPkts        = 0

    for pcapFileName in pcapFileNames:
        relativePathToInputFile = f'{tracePath}/{pcapFileName}'       
        checkIfInputFileExists (relativePathToInputFile)    
        filePktNum = 0
        
        tic ()
    
        # Open the pcap file for reading pkts one by one and the output file for writing hashes
        with PcapReader(relativePathToInputFile) as pkts:
            
            # Loop through pkts one by one
            for pkt in pkts:
                if totalPkts >= maxNumOfPkts:
                    break  # Stop if max number of pkts has been reached
    
                try:
    
                    if pkt.haslayer('IP'):
                        # Handle TCP pkts
                        srcIp, dstIp     = pkt['IP'].src, pkt['IP'].dst
                        
                        if pkt.haslayer('TCP'):
                            srcPort, dstPort = pkt['TCP'].sport, pkt['TCP'].dport
                        elif pkt.haslayer('UDP'):
                            srcPort, dstPort = pkt['UDP'].sport, pkt['UDP'].dport
                        else:
                            # For non-TCP/UDP pkts with IP layer, ignore the ports
                            nonTcpUdpIpPkts += 1  # Increment non-TCP/UDP IP packet count
                            if not(parseNonTcpUdpIpPkts):
                                continue
                            srcPort, dstPort = 0, 0
                    else:
                        # For non-IP pkts, set IP and port info to default values
                        nonIpPkts += 1  # Increment non-IP packet count
                        if not(parseNonIpPkts):
                            continue
                        srcPort, dstPort = 0, 0
                        srcIp, dstIp = '0.0.0.0', '0.0.0.0'
    
                    # Write the hash to the output file
                    printf (outputFile, '{}\n' .format(
                        mmh3.hash(f'{srcPort}-{dstPort}-{srcIp}-{dstIp}', signed=True)
                    ))
                    
                    totalPkts  += 1  # Increment total packet count
                    filePktNum += 1
    
                except Exception as e:
                    print(f"Error processing packet {pktNum}: {e}")
                    continue  # Skip any packet that raises an exception

        # Output the count for this file
        print(f'{genElapsedTimeStr (toc())}. Finished parsing {filePktNum} pkts from file {pcapFileName}')

    # Output the final counts
    print(f'Finished parsing {totalPkts} pkts, from which {nonTcpUdpIpPkts} are IP but non-TCP/UDP, and {nonIpPkts} are non-IP.')

# maxNumOfPkts = 100000000
# processPcapAndWriteHashes (
#     outputFileName  = 'Caida1_equinix-nyc.dirA.20181220-130000.UTC.anon.txt', 
#     pcapFileNames   = ['Caida1_equinix-nyc.dirA.20181220-130000.UTC.anon.pcap',
#                        'Caida1_equinix-nyc.dirA.20181220-130100.UTC.anon.pcap',
#                        'Caida1_equinix-nyc.dirA.20181220-130200.UTC.anon.pcap'],
#     maxNumOfPkts    = maxNumOfPkts
# )
# maxNumOfPkts = 100000000
# processPcapAndWriteHashes (
#     outputFileName  = 'Caida2_equinix-chicago.dirA.20160406-130000.UTC.anon.txt', 
#     pcapFileNames   = ['Caida2_equinix-chicago.dirA.20160406-130000.UTC.anon.pcap',
#                         'Caida2_equinix-chicago.dirA.20160406-130100.UTC.anon.pcap',
#                         'Caida2_equinix-chicago.dirA.20160406-130200.UTC.anon.pcap',
#                         'Caida2_equinix-chicago.dirA.20160406-130300.UTC.anon.pcap'],
#     maxNumOfPkts    = maxNumOfPkts
# )

a = np.empty([0], dtype='int32')
print (a)