import matplotlib 
import matplotlib.pyplot as plt
import math, random, os, pickle, mmh3, time, csv, numpy as np
import itertools
from datetime import datetime

import settings
from ttictoc import tic,toc
from printf import printf, printarFp
from settings import *  

def condenseTrace (
        maxNumRows      = float('inf'), # overall number of increments (# of pkts in the trace) 
        traceFileName   = None,
    ):
    """
    Parse a trace. Collect stat about the trace.
    The trace is merely a list of integers (keys), representing the flowId to which each pkt belongs.
    Optional outputs:
    - print the trace's stat to a file.
    - print to file an compressed version of the trace, where each key is replaced by a unique flowId, allocated to each flow in sequential order of appearance in the trace. 
    """
    relativePathToInputFile = getRelativePathToTraceFile (f'{traceFileName}.csv')
    checkIfInputFileExists (relativePathToInputFile)
    trace           = np.fromfile(relativePathToInputFile, count = maxNumRows, sep='\n', dtype=FLOW_TYPE)
    condensedTrace  = np.empty (maxNumRows, dtype=FLOW_TYPE)
    listOfFlowKeys  = np.empty([0], dtype=FLOW_TYPE)
    flowSizes       = np.empty([0], dtype=FLOW_TYPE)
    
    tic ()
    pktNum = 0
    print (f'Started parsing trace {traceFileName}.csv')        
    for flowKey in trace:
        flowId = np.where(listOfFlowKeys==flowKey)[0]
        if (len(flowId)==0): # first pkt of this flow
            flowId = len(flowSizes)
            listOfFlowKeys = np.append (listOfFlowKeys, flowKey)
            flowSizes = np.append (flowSizes, 1) # the size of this flow is 1
        elif (len(flowId)==2):
            error (f'In TraceParser.condenseTrace(). FlowId {flowId} is duplicated in listOfFlowKeys')
        else: # the pkt belongs to a known flow 
            flowId = flowId[0]
            flowSizes[flowId] += 1
            condensedTrace[pktNum] = flowId
        pktNum += 1
            
    relativePathToTraceOutputFile = relativePathToInputFile.split('.txt')[0] + '_condensed.txt'
    np.savetxt(relativePathToTraceOutputFile, condensedTrace[:pktNum], fmt='%d')
    printf (traceOutputFile, f'{flowId}\n')

    print (f'Finished parsing {traceFileName}.txt after {genElapsedTimeStr(toc())}. num of flows={flowSizes.shape[0]}')

def calcDenseTraceStat (
        traceFileName = None,
        maxNumOfRows  = INF_INT,
        numFlows      = 2000000
        ):
    """
    Collect stat about the trace, and print it to a file.
    The trace is merely a list of integers (keys), representing the flow to which each pkt belongs, in a .txt file.
    The trace must be condensed, namely, the flowIds are 0, ..., flowId-1
    """
    relativePathToTraceFile = getRelativePathToTraceFile (f'{traceFileName}.txt')
    checkIfInputFileExists (relativePathToTraceFile)
    if numFlows==None:
        error ('In TraceParser.calcDenseTraceStat(). Sorry, currently you must specify the num of flows for parsing the trace.')
    traceFile = open (relativePathToTraceFile, 'r')
    trace = np.fromfile(relativePathToTraceFile, count = maxNumOfRows, sep='\n', dtype='uint32')

    flowSizes                 = np.zeros (numFlows,     dtype=FLOW_TYPE)
    interAppearanceVec        = np.zeros (maxNumOfRows, dtype=FLOW_TYPE)
    last_appearance_of        = np.zeros (numFlows,     dtype=FLOW_TYPE)
    idx_in_interAppearanceVec = 0
    maxNumOfRows = min(maxNumOfRows, len(trace))
    for rowNum in range(maxNumOfRows):
        flowId = trace[rowNum]
        flowSizes[flowId] += 1        
        if last_appearance_of[flowId]>0: # This key has already appeared before 
            interAppearanceVec[idx_in_interAppearanceVec] = rowNum-last_appearance_of[flowId]
            idx_in_interAppearanceVec += 1 
        last_appearance_of[flowId] = rowNum
        
    interAppearanceVec = interAppearanceVec[:idx_in_interAppearanceVec]
    flowSizes = flowSizes[np.where(flowSizes>0)[0]].astype(FLOW_TYPE)
    printTraceStatToFile (
        traceFileName       = traceFileName, 
        flowSizes           = flowSizes,
        interAppearanceVec  = interAppearanceVec
    )
        
        
def printTraceStatToFile (
    interAppearanceVec  : np.array,
    flowSizes           : np.array,
    traceFileName       : str,
    ):
    """
    Given a vector with the flowId accessed at each cycle, calculate the trace's stat.
    """    
    relativePathToStatFile = getRelativePathToTraceFile (f'{traceFileName}_stat.txt')
    statFile    = open (relativePathToStatFile, 'w')
    if len(interAppearanceVec)==0:
        error ('In TraceParser.printTraceStatToFile(). interAppearanceVec is empty. Did you use too short sim?')
    printf (statFile, '// mean inter arrival = {:.2f}\n'  .format(np.mean(interAppearanceVec)))
    printf (statFile, '// stdev inter arrival = {:.2f}\n' .format(np.std(interAppearanceVec)))
    writeVecStatToFile (
        statFile    = statFile,
        vec         = flowSizes,
        str         = 'flow sizes'        
    )
    
calcDenseTraceStat (
    traceFileName   = getTraceFullName('Caida1'),
    maxNumOfRows    = 25000000,
)

# condenseTrace (
#     traceFileName   = 'Caida2_equinix-nyc.dirA.20181220-130000.UTC.anon',
#     maxNumRows      = 25000000 
# )
