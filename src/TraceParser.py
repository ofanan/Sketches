import matplotlib 
import matplotlib.pyplot as plt
import math, random, os, pickle, mmh3, time, csv, numpy as np
import itertools
from datetime import datetime

import settings
from   tictoc import tic, toc # my modules for measuring and print-out the simulation time.
from printf import printf, printarFp
from settings import warning, error, indexOrNone, getTracesPath

def parseCsvTrace (
        maxNumRows      = float('inf'), # overall number of increments (# of pkts in the trace) 
        traceFileName   = None,
        verbose         = [] # verbose level, determined in settings.py.
        ):
    """
    Parse a trace. Collect stat about the trace.
    The trace is merely a list of integers (keys), representing the flow to which each pkt belongs.
    Optional outputs:
    - print the trace's stat to a file.
    - print to file an compressed version of the trace, where each key is replaced by a unique flowId, allocated to each flow in sequential order of appearance in the trace. 
    """
    relativePathToInputFile = settings.getRelativePathToTraceFile (traceFileName, exitError=True)
    csvFile = open (relativePathToInputFile, 'r')
    csvReader = csv.reader(csvFile) 

    rowNum          = 1
    listOfFlowKeys  = []
    flowSizes       = []
    
    if settings.VERBOSE_RES in verbose:
        relativePathToTraceOutputFile = relativePathToInputFile.split('.csv')[0] + '.txt'
        traceOutputFile = open (relativePathToTraceOutputFile, 'w')
        
    for row in csvReader:
        flowKey = int(row[0])
        flowId = indexOrNone(l=listOfFlowKeys, elem=flowKey)
        if flowId==None: # first pkt of this flow
            flowId = len(flowSizes)
            listOfFlowKeys.append (flowKey)
            flowSizes.append (1) # the size of this flow is 1
        else:
            flowSizes[flowId] += 1
        if settings.VERBOSE_RES in verbose:
            printf (traceOutputFile, f'{flowId}\n')
        rowNum += 1
        if rowNum > maxNumRows:
            break

    statFile    = open (f'../res/{traceFileName}_stat.txt', 'w')
    numFlows    = len(flowSizes)
    printf (statFile, f'// numFlows = {numFlows}\n')
    printf (statFile, f'// flowSizes={flowSizes}\n')
    printf (statFile, f'// maxFlowSize={max(flowSizes)}\n')
    # maxFlowSize = max (flowSizes)
    # numBins = min (100, maxFlowSize+1)
    # binSize = maxFlowSize // (numBins-1)
    # binVal  = [None] * numBins 
    # for bin in range(numBins):
    #     binVal[bin] = len ([flowId for flowId in range(numFlows) if (flowRealVal[flowId]//binSize)==bin])
    # binFlowSizes = [binSize*bin for bin in rangum zero flows={len ([item for item in flowRealVal if item==0])}, num non-zeros flows={len ([item for item in flowRealVal if item>0])}')
    # printf (outputFile, f'\nmaxFlowSize={maxFlowSize}, binVal={binVal}')
    # printf (outputFile, f'\nbinFlowSizes={binFlowSizes}')
    # printf (outputFile, f'\nflowSizes={flowRealVal}')
    # _, ax = plt.subplots()
    # ax.plot ([binSize*bin for bin in range (numBins)], binVal)
    # ax.set_yscale ('log')
    # plt.savefig (f'../res/{outputFileName}.pdf', bbox_inches='tight')        


def calcTraceStat (
        traceFileName = None,
        maxNumOfRows  = float('inf'),
        numFlows      = None
        ):
    """
    Collect stat about the trace, and print it to a file.
    The trace is merely a list of integers (keys), representing the flow to which each pkt belongs, in a .txt file.
    """
    relativePathToTraceFile = settings.getRelativePathToTraceFile (traceFileName)
    if numFlows==None:
        error ('In TraceParser.calcTraceStat(). Sorry, currently you must specify the num of flows for parsing the trace.')
    settings.checkIfInputFileExists (relativePathToTraceFile)
    traceFile = open (relativePathToTraceFile, 'r')

    rowNum = 0
    flowSizes = [0]*numFlows
    for row in traceFile:            
        rowNum += 1
        flowSizes[int(row)] += 1        
        if rowNum>maxNumOfRows:
            break 
        
    statFile    = open (f'../res/{traceFileName}_stat.txt', 'a+')
    numFlows    = len(flowSizes)
    maxFlowSize = max(flowSizes)
    printf (statFile, f'// numFlows = {numFlows}\n')
    printf (statFile, f'// flowSizes={flowSizes}\n')
    printf (statFile, f'// maxFlowSize={maxFlowSize}\n')
    
    numBins = min (100, maxFlowSize+1)
    binSize = maxFlowSize // (numBins-1)
    binVal  = [None] * numBins 
    for bin in range(numBins):
        binVal[bin] = len ([flowId for flowId in range(numFlows) if (flowSizes[flowId]//binSize)==bin])
    binFlowSizes = [binSize*bin for bin in range(numBins)]
    printf (statFile, f'// bins:\n')
    for bin in range(numBins):
        printf (statFile, f'binFlowSizes={binFlowSizes[bin]}, binVal={binVal[bin]}\n')
    statFile.close()
    
# parseCsvTrace (
#     traceFileName = 'Caida2.csv',
#     verbose         = [settings.VERBOSE_RES] # verbose level, determined in settings.py.
# )

calcTraceStat (traceFileName = 'Caida2.txt', numFlows = 10000000)
