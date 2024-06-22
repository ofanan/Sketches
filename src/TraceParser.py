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

    printTraceStatToFile (traceFileName=traceFileName, flowSizes=flowSizes)

def calcTraceStat (
        traceName     = None,
        maxNumOfRows  = float('inf'),
        numFlows      = 2000000
        ):
    """
    Collect stat about the trace, and print it to a file.
    The trace is merely a list of integers (keys), representing the flow to which each pkt belongs, in a .txt file.
    """
    relativePathToTraceFile = settings.getRelativePathToTraceFile (f'{traceName}.txt')
    if numFlows==None:
        error ('In TraceParser.calcTraceStat(). Sorry, currently you must specify the num of flows for parsing the trace.')
    traceFile = open (relativePathToTraceFile, 'r')

    flowSizes                 = np.zeros (numFlows,     dtype='int32')
    interAppearanceVec        = np.zeros (maxNumOfRows, dtype='int32')
    last_appearance_of        = np.zeros (numFlows,     dtype='int32')
    idx_in_interAppearanceVec = 0
    rowNum                      = 0
    for row in traceFile:            
        rowNum += 1
        flowId = int(row)
        flowSizes[flowId] += 1        
        if last_appearance_of[flowId]>0: # This key has already appeared before #is the first appearance of this key
            interAppearanceVec[idx_in_interAppearanceVec] = rowNum-last_appearance_of[flowId]
            idx_in_interAppearanceVec += 1 
        last_appearance_of[flowId] = rowNum
        if rowNum>maxNumOfRows:
            break 
        
    interAppearanceVec = interAppearanceVec[:idx_in_interAppearanceVec]        
    printTraceStatToFile (
        traceName           = traceName, 
        flowSizes           = flowSizes,
        interAppearanceVec  = interAppearanceVec
    )
        
def printTraceStatToFile (
    traceName       = None,
    flowSizes           = [],
    interAppearanceVec  = None
    ):
    """
    Given a vector with the flowId accessed at each cycle, calculate the trace's stat.
    """    
    statFile    = open (settings.getRelativePathToTraceFile (f'{traceName}_stat.txt', checkIfFileExists=False), 'w')
    flowSizes   = np.array([f for f in flowSizes if f>0])
    numFlows    = len(flowSizes)
    maxFlowSize = max(flowSizes)
    printf (statFile, f'// numFlows = {numFlows}\n')
    printf (statFile, '// mean inter arrival = {:.2e}\n'  .format(np.mean(interAppearanceVec)))
    printf (statFile, '// stdev inter arrival = {:.2e}\n' .format(np.std(interAppearanceVec)))
    printf (statFile, f'// maxFlowSize={maxFlowSize}\n')
    printf (statFile, f'// avgFlowSize={np.mean(flowSizes)}\n') 
    printf (statFile, f'// stdevFlowSize={np.std(flowSizes)}\n')
    
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
#     traceFileName = 'Caida1.csv',
#     verbose         = [settings.VERBOSE_RES] # verbose level, determined in settings.py.
# )

traceName = '‏‏Caida1'
calcTraceStat (
    traceName     = traceName, 
    maxNumOfRows  = 25000000,
)
