from scapy.all import rdpcap
import numpy as np
import mmh3

def processPcap(filePath, maxNumOfPkts):
    # Read up to maxNumOfPkts packets from the pcap file
    packets = rdpcap(filePath, count=maxNumOfPkts)

    # Determine the number of tuples (this will be min of the number of packets read and maxNumOfPkts)
    numTuples = len(packets)
    
    # Initialize a NumPy array to store the 4-tuple information
    tuples = np.empty((numTuples, 4), dtype=object)

    # Extract the 4-tuple (source port, dest port, source IP, dest IP) for each packet
    pktNum = 0

    for pkt in packets:
        if pkt.haslayer('IP'):
            if pkt.haslayer('TCP'):
                tuples[pktNum] = (pkt['TCP'].sport, pkt['TCP'].dport, pkt['IP'].src, pkt['IP'].dst)
                pktNum += 1
            elif pkt.haslayer('UDP'):
                tuples[pktNum] = (pkt['UDP'].sport, pkt['UDP'].dport, pkt['IP'].src, pkt['IP'].dst)
                pktNum += 1

    # Define the hashing function using a lambda
    hashTuple = lambda t: mmh3.hash(f"{t[0]}-{t[1]}-{t[2]}-{t[3]}", signed=True)

    # Vectorize the hashing function
    vectorizedHash = np.vectorize(hashTuple, otypes=[np.int32])

    # Apply the vectorized hash function to the tuples array
    hashArray = vectorizedHash(tuples[:pktNum])

    return hashArray

# Example usage
maxNumOfPkts = 2  # Limit the number of packets read
hashedVector = processPcap('../../traces/Caida/Caida1_equinix-nyc.dirA.20181220-130000.UTC.anon.pcap', maxNumOfPkts)
print(hashedVector)
