"""
Parses a .pcap file. 
"""
from scapy.all import *
import numpy as np, mmh3, csv
from ttictoc import tic,toc
import pandas as pd
import settings
from settings import *

def processPcapAndWriteHashes(
        traceFileName,      
        maxNumOfPkts    = INF_INT, # maximum number of pkts to be parsed, starting from the beginning of the trace
    ):
    """
    Parse a .pcap file. Write the parsed file to a .csv file. bearing the same fileName as the .pcap file, but with extension .csv instead of .pcap. 
    """

    tracePath = getTracesPath() + 'Caida' 
    relativePathToInputFile = f'{tracePath}/{traceFileName}'       
    checkIfInputFileExists (relativePathToInputFile)    

    relativePathToOutputFile = f'{tracePath}/{traceFileName}.txt'
    # Initialize counters
    totalPackets        = 0
    nonTcpUdpIpPackets  = 0
    nonIpPackets        = 0

    tic ()
    # Open the pcap file for reading packets one by one and the output file for writing hashes
    with PcapReader(relativePathToInputFile) as packets, open(relativePathToOutputFile, 'w') as outputFile:
        pktNum = 0
        
        # Loop through packets one by one
        for pkt in packets:
            if maxNumOfPkts is not None and pktNum >= maxNumOfPkts:
                break  # Stop if max number of packets has been reached

            try:
                totalPackets += 1  # Increment total packet count

                if pkt.haslayer('IP'):
                    # Handle TCP packets
                    if pkt.haslayer('TCP'):
                        srcPort, dstPort = pkt['TCP'].sport, pkt['TCP'].dport
                        srcIp, dstIp = pkt['IP'].src, pkt['IP'].dst
                    # Handle UDP packets
                    elif pkt.haslayer('UDP'):
                        srcPort, dstPort = pkt['UDP'].sport, pkt['UDP'].dport
                        srcIp, dstIp = pkt['IP'].src, pkt['IP'].dst
                    else:
                        # For non-TCP/UDP packets with IP layer, ignore the ports
                        srcPort, dstPort = 0, 0
                        srcIp, dstIp = pkt['IP'].src, pkt['IP'].dst
                        nonTcpUdpIpPackets += 1  # Increment non-TCP/UDP IP packet count
                else:
                    # For non-IP packets, set IP and port info to default values
                    srcPort, dstPort = 0, 0
                    srcIp, dstIp = '0.0.0.0', '0.0.0.0'
                    nonIpPackets += 1  # Increment non-IP packet count

                # Create the 4-tuple and hash it
                fourTuple = (srcPort, dstPort, srcIp, dstIp)
                hashedValue = mmh3.hash(f"{fourTuple[0]}-{fourTuple[1]}-{fourTuple[2]}-{fourTuple[3]}", signed=True)

                # Write the hash to the output file
                outputFile.write(f"{hashedValue}\n")
                
                pktNum += 1

            except Exception as e:
                print(f"Error processing packet {pktNum}: {e}")
                continue  # Skip any packet that raises an exception

    # Output the final counts
    print(f'{genElapsedTimeStr (toc())}. Finished parsing {totalPackets} packets, from which {nonTcpUdpIpPackets} are IP but non-TCP/UDP, and {nonIpPackets} are non-IP.')
    # print (f'genElapsedTimeStr (toc()). Finished parsing {pktNum} pkts by multirow after {time.time() - startTime} sec')

maxNumOfPkts = 100000000
processPcapAndWriteHashes (
    # traceFileName='Caida1_equinix-nyc.dirA.20181220-130000.UTC.anon.pcap', 
    traceFileName='Caida2_equinix-chicago.dirA.20160406-130000.UTC.anon.pcap', 
    maxNumOfPkts=maxNumOfPkts
)
