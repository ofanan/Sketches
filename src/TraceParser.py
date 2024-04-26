import matplotlib 
import matplotlib.pyplot as plt
import math, random, os, pickle, mmh3, time, csv, numpy as np
from datetime import datetime

import settings
from   tictoc import tic, toc # my modules for measuring and print-out the simulation time.
from printf import printf, printarFp
from settings import warning, error

def parseTrace (
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
    relativePathToInputFile = settings.getRelativePathToTraceFile (traceFileName)
    settings.checkIfInputFileExists (relativePathToInputFile)
    csvFile = open (relativePathToInputFile, 'r')
    csvReader = csv.reader(csvFile) 

    rowNum              = 1
    flowsListOfDicts    = []
    flowId              = int(0)

    if settings.VERBOSE_RES in verbose:
        relativePathToTraceOutputFile = relativePathToInputFile.split('.csv')[0] + '.txt'
        traceOutputFile = open (relativePathToTraceOutputFile, 'w')
        
    for row in csvReader:
        flowKey = int(row[0]) 
        dictsWithThisFlowKey = [item for item in flowsListOfDicts if item['key']==flowKey]
        if len(dictsWithThisFlowKey)>1:
            error ('len(dictsWithThisFlowKey)={len(dictsWithThisFlowKey)}, flowKey={flowKey}')
        elif len(dictsWithThisFlowKey)==1: # This flow already appeared --> inc. its cnt.
            dictsWithThisFlowKey[0]['cnt'] += 1
            if settings.VERBOSE_RES in verbose:
                printf (traceOutputFile, '{}\n' .format(dictsWithThisFlowKey[0]['id']))
        else: # This flow hasn't appeared yet --> insert it into flowsListOfDicts
            flowsListOfDicts.append (
                {'key' : flowKey,
                 'id'  : flowId,
                 'cnt' : 1}
            )
            if settings.VERBOSE_RES in verbose:
                printf (traceOutputFile, f'{flowId}\n')
            flowId += 1
        rowNum += 1
        if rowNum > maxNumRows:
            break

    flowSizes   = [item['cnt'] for item in flowsListOfDicts]
    statFile    = open (f'../res/{traceFileName}_stat.txt', 'w')
    numFlows    = len(flowSizes)
    printf (statFile, f'// numFlows = {numFlows}\n')
    printf (statFile, '// flowSizes={flowSizes}\n')
    
    
        
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

    
parseTrace (
    # maxNumRows      = 10000, #float('inf'), # overall number of increments (# of pkts in the trace) 
    traceFileName   = 'Caida1',
    verbose         = [settings.VERBOSE_RES] # verbose level, determined in settings.py.
)
