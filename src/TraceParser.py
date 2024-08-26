import matplotlib 
import matplotlib.pyplot as plt
import math, random, os, pickle, mmh3, time, csv, numpy as np
import itertools
from datetime import datetime

import settings
from ttictoc import tic,toc
from printf import printf, printarFp
from settings import *  

def parseCsvTrace (
        maxNumRows      = float('inf'), # overall number of increments (# of pkts in the trace) 
        traceFileName   = None,
        verbose         = [VERBOSE_RES] # verbose level, determined in settings.py.
        ):
    """
    Parse a trace. Collect stat about the trace.
    The trace is merely a list of integers (keys), representing the flow to which each pkt belongs.
    Optional outputs:
    - print the trace's stat to a file.
    - print to file an compressed version of the trace, where each key is replaced by a unique flowId, allocated to each flow in sequential order of appearance in the trace. 
    """
    relativePathToInputFile = getRelativePathToTraceFile (f'{traceFileName}.csv')
    checkIfInputFileExists (relativePathToInputFile)
    trace           = np.fromfile(relativePathToInputFile, count = self.maxNumIncs, sep='\n', dtype='uint32')
    listOfFlowKeys  = np.empty(dtype=FLOW_TYPE)
    flowSizes       = np.empty(dtype=FLOW_TYPE)
    
    if VERBOSE_RES in verbose:
        relativePathToTraceOutputFile = relativePathToInputFile.split('.csv')[0] + '.txt'
        traceOutputFile = open (relativePathToTraceOutputFile, 'w')
    else:
        warning ('TraceParser.parseCsvTrace() was called without VERBOSE_RES')

    tic ()
    print (f'Started parsing trace {traceFileName}.csv')        
    for flowKey in trace:
        flowId = np.where(listOfFlowKeys==flowKey)[0]
        if (len(flowId)==0): # first pkt of this flow
            flowId = len(flowSizes)
            listOfFlowKeys = np.append (listOfFlowKeys, flowKey)
            flowSizes = np.append (flowSizes, 1) # the size of this flow is 1
        if (len(flowId)==2):
            error (f'In TraceParser.parseCsvTrace(). FlowId {} is duplicated in listOfFlowKeys')
        else:
            flowId = flowId[0]
            flowSizes[flowId] += 1
        if VERBOSE_RES in verbose:
            printf (traceOutputFile, f'{flowId}\n')

    print (f'Finished parsing after {genElapsedTimeStr(toc())}')

def calcTraceStat (
        traceFileName = None,
        maxNumOfRows  = INF_INT,
        numFlows      = 2000000
        ):
    """
    Collect stat about the trace, and print it to a file.
    The trace is merely a list of integers (keys), representing the flow to which each pkt belongs, in a .txt file.
    """
    relativePathToTraceFile = getRelativePathToTraceFile (f'{traceFileName}.txt')
    checkIfInputFileExists (relativePathToTraceFile)
    if numFlows==None:
        error ('In TraceParser.calcTraceStat(). Sorry, currently you must specify the num of flows for parsing the trace.')
    traceFile = open (relativePathToTraceFile, 'r')
    trace = np.fromfile(relativePathToInputFile, count = self.maxNumIncs, sep='\n', dtype='uint32')

    flowSizes                 = np.zeros (numFlows,     dtype=FLOW_TYPE)
    interAppearanceVec        = np.zeros (maxNumOfRows, dtype=FLOW_TYPE)
    last_appearance_of        = np.zeros (numFlows,     dtype=FLOW_TYPE)
    idx_in_interAppearanceVec = 0
    for rowNum in range(maxNumOfRows):
        flowId = trace[rowNum]
        flowSizes[flowId] += 1        
        if last_appearance_of[flowId]>0: # This key has already appeared before #is the first appearance of this key
            interAppearanceVec[idx_in_interAppearanceVec] = rowNum-last_appearance_of[flowId]
            idx_in_interAppearanceVec += 1 
        last_appearance_of[flowId] = rowNum
        
    interAppearanceVec = interAppearanceVec[:idx_in_interAppearanceVec]        
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
    
# calcTraceStat (
#     traceFileName = 'Caida1_equinix-chicago.dirA.20160406-130000.UTC.anon'
# )

parseCsvTrace (
    traceFileName   = 'Caida2_equinix-nyc.dirA.20181220-130000.UTC.anon.pcap',
    maxNumRows      = 25 #$$$$ 000000,
)
