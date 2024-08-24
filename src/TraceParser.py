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
    relativePathToInputFile = getRelativePathToTraceFile (traceFileName)
    checkIfInputFileExists (relativePathToInputFile)
    csvFile = open (relativePathToInputFile, 'r')
    csvReader = csv.reader(csvFile) 

    rowNum          = 1
    listOfFlowKeys  = []
    flowSizes       = []
    
    if VERBOSE_RES in verbose:
        relativePathToTraceOutputFile = relativePathToInputFile.split('.csv')[0] + '.txt'
        traceOutputFile = open (relativePathToTraceOutputFile, 'w')
    else:
        warning ('TraceParser.parseCsvTrace() was called without VERBOSE_RES')

    tic ()
    print (f'Started parsing trace {traceFileName}')        
    for row in csvReader:
        flowKey = int(row[0])
        flowId = indexOrNone(l=listOfFlowKeys, elem=flowKey)
        if flowId==None: # first pkt of this flow
            flowId = len(flowSizes)
            listOfFlowKeys.append (flowKey)
            flowSizes.append (1) # the size of this flow is 1
        else:
            flowSizes[flowId] += 1
        if VERBOSE_RES in verbose:
            printf (traceOutputFile, f'{flowId}\n')
        rowNum += 1
        if rowNum > maxNumRows:
            break

    printTraceStatToFile (traceFileName=traceFileName, flowSizes=flowSizes)
    print (f'Finished parsing after {genElapsedTimeStr(toc())}')

def calcTraceStat (
        traceName     = None,
        maxNumOfRows  = float('inf'),
        numFlows      = 2000000
        ):
    """
    Collect stat about the trace, and print it to a file.
    The trace is merely a list of integers (keys), representing the flow to which each pkt belongs, in a .txt file.
    """
    relativePathToTraceFile = getRelativePathToTraceFile (f'{traceName}.txt')
    checkIfInputFileExists (relativePathToTraceFile)
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
    relativePathToStatFile = getRelativePathToTraceFile (f'{traceName}_stat.txt')
    checkIfInputFileExists (relativePathToStatFile)
    statFile    = open (relativePathToStatFile, 'w')
    printf (statFile, '// mean inter arrival = {:.2f}\n'  .format(np.mean(interAppearanceVec)))
    printf (statFile, '// stdev inter arrival = {:.2f}\n' .format(np.std(interAppearanceVec)))
    writeVecStatToFile (
        statFile    = statFile,
        vec         = np.array([f for f in flowSizes if f>0]),
        str         = 'flow sizes'        
    )
    
parseCsvTrace (
    traceFileName   = 'Caida1.csv',
    maxNumRows      = 25000000,
)

# traceName = 'Caida1'
# calcTraceStat (
#     traceName     = traceName, 
#     maxNumOfRows  = 250, #$$$$ 00000,
# )

# relativePathToInputFile = '../../traces/Caida/‏‏Caida1.txt'
# if os.path.isfile (relativePathToInputFile):
#     error ('yesh')
# error ('ein')
