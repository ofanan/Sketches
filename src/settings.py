# Parameters and accessory functions
# import math, random, os, pandas as pd
import os, math, numpy as np, scipy.stats as st 
from printf import printf

SEED    = 42
INF_INT = 999999999
MAX_NUM_OF_FLOWS = 2**32

VERBOSE_COUT_CONF       = 0 # print to stdout details about the configuration, e.g., cntrSize, hyperSize, Vmax, bias.
VERBOSE_COUT_CNTRLINE   = 1 # print to stdout details about the concrete counter and its fields.
VERBOSE_DEBUG           = 2 # perform checks and debug operations during the run.
VERBOSE_RES             = 3 # print output to a .res file in the directory ../res
VERBOSE_DETAILED_RES    = 4
VERBOSE_FULL_RES        = 5
VERBOSE_PCL             = 6 # print output to a .pcl file in the directory ../res/pcl_files
VERBOSE_DETAILS         = 7 # print to stdout details about the counter
VERBOSE_NOTE            = 8 # print to stdout notes, e.g. when the target cntr value is above its max or below its min.
VERBOSE_LOG             = 9
VERBOSE_LOG_END_SIM     = 10
VERBOSE_DETAILED_LOG    = 11
VERBOSE_PROGRESS        = 12 # Print periodical output notifying the progress. Used to control long runs.
VERBOSE_LOG_CNTRLINE    = 14  

# Configurations to be run. 
# For cntrSize<8, the conf' the values are unrealistically small, and used only for checks and debugging.
# For cntrSize>=8, cntrMaxVal is calculated by that reached by F2P stat, and hyperSize is the corresponding hyper-exponent field size in F2P stat.
# expSize is the minimal needed for SEAD stat to reach the requested value.
Confs = [{'cntrSize' : 3,  'cntrMaxVal' : 10,       'hyperSize' : 2, 'hyperMaxSize' : 1, 'seadExpSize' : 1},
         {'cntrSize' : 4,  'cntrMaxVal' : 22,       'hyperSize' : 2, 'hyperMaxSize' : 1, 'seadExpSize' : 1},
         {'cntrSize' : 6,  'cntrMaxVal' : 31744,    'hyperSize' : 2, 'hyperMaxSize' : 1, 'seadExpSize' : 4},
         {'cntrSize' : 7,  'cntrMaxVal' : 64512,    'hyperSize' : 2, 'hyperMaxSize' : 1, 'seadExpSize' : 4},
         {'cntrSize' : 8,  'cntrMaxVal' : 130048,   'hyperSize' : 2, 'hyperMaxSize' : 3, 'seadExpSize' : 4},
         {'cntrSize' : 9,  'cntrMaxVal' : 261120,   'hyperSize' : 2, 'hyperMaxSize' : 3, 'seadExpSize' : 4},
         {'cntrSize' : 10, 'cntrMaxVal' : 523264,   'hyperSize' : 2, 'hyperMaxSize' : 3, 'seadExpSize' : 4},
         {'cntrSize' : 11, 'cntrMaxVal' : 1047552,  'hyperSize' : 2, 'hyperMaxSize' : 3, 'seadExpSize' : 4},
         {'cntrSize' : 12, 'cntrMaxVal' : 2096128,  'hyperSize' : 2, 'hyperMaxSize' : 3, 'seadExpSize' : 4},
         {'cntrSize' : 13, 'cntrMaxVal' : 4193280,  'hyperSize' : 2, 'hyperMaxSize' : 3, 'seadExpSize' : 4},
         {'cntrSize' : 14, 'cntrMaxVal' : 8387584,  'hyperSize' : 2, 'hyperMaxSize' : 3, 'seadExpSize' : 4},
         {'cntrSize' : 15, 'cntrMaxVal' : 16776192, 'hyperSize' : 2, 'hyperMaxSize' : 3, 'seadExpSize' : 4},
         {'cntrSize' : 16, 'cntrMaxVal' : 33553408, 'hyperSize' : 2, 'hyperMaxSize' : 3, 'seadExpSize' : 4}]

# Calculate the confidence interval of an array of values ar, given its avg. Based on 
# https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
confInterval = lambda ar, avg, conf_lvl=0.99 : st.t.interval (conf_lvl, len(ar)-1, loc=avg, scale=st.sem(ar)) if np.std(ar)>0 else [avg, avg]
   
def getConfByCntrSize (cntrSize):
    """
    given the counter's size, return the configuration with that counter size.
    If the number of configurations with that counter's size, exit with a proper error message.
    """
    listOfConfs = [item for item in Confs if item['cntrSize']==cntrSize]
    if (len(listOfConfs)<1): 
        error (f'Sorry. No known configuration for cntrSize={cntrSize}')
    elif (len(listOfConfs)>1):
        error (f'Sorry. Too many known configurations for cntrSize={cntrSize}')
    return listOfConfs[0]
   
def getCntrMaxValByCntrSize (cntrSize):
    """
    given the counter's size, return the counter's max size of the (single) configuration with that counter size.
    If the number of configurations with that counter's size, exit with a proper error message.
    """
    return getConfByCntrSize (cntrSize)['cntrMaxVal']
   
   
def idxOfLeftmostZero (ar, maxIdx):
    """
    if the index of the leftmost 0 in the array >= maxIdx, return maxIdx.
    else, return the index of the leftmost 0 in the array.
    """ 
    if (ar == '1' * len(ar)): 
        return maxIdx
    return min (ar.index('0'), maxIdx)
    
def checkCntrIdx (cntrIdx, numCntrs, cntrType):
    """
    Check if the given cntr index is feasible.
    If not - print error msg and exit.
    """
    if (cntrIdx < 0 or cntrIdx>(numCntrs-1)):
        print ('error in {}: wrong cntrIdx. Please select cntrIdx between 0 and {}' .format (cntrType, numCntrs-1))
        exit ()
    
def sortcntrMaxVals ():
    """
    Read the file '../res/cntrMaxVals.txt". Sort it in an increasing fashion of the max cntr vals.
    Print the results to '../res/maxC
    """
    input_file      = open ('../res/cntrMaxVals.txt', 'r')
    lines           = (line.rstrip() for line in input_file) # "lines" contains all lines in input file
    lines           = (line for line in lines if line)       # Discard blank lines
    
    list_of_dicts = []
    for line in lines:

        # Discard lines with comments / verbose data
        if (line.split ("//")[0] == ""):
            continue
        
        splitted_line = line.split ()
        if len(splitted_line)==0:
            error ('in settings.sortcntrMaxVals(). line={line}, splitted_line={splitted_line}')
        mode = splitted_line[0]
        list_of_dict = [item['mode'] for item in list_of_dicts if item['mode']==mode]
        if (mode not in [item['mode'] for item in list_of_dicts if item['mode']==mode]):
            if len(splitted_line)<1:
                error ('in settings.sortcntrMaxVals(). line={line}, splitted_line={splitted_line}')
            if len(splitted_line[1])<1:
                error ('in settings.sortcntrMaxVals(). line={line}, splitted_line={splitted_line}')
            list_of_dicts.append ({'mode' : mode, 'cntrSize' : int(mode.split('_n')[1].split('_')[0]), 'maxVal' : float(splitted_line[1].split('=')[1])})
    list_of_dicts =  sorted (list_of_dicts, key = lambda item : (item['cntrSize'], item['maxVal']))
    output_file   = open ('../res/cntrMaxValsSorted.txt', 'w')
    for item in list_of_dicts:
        val = item['maxVal']
        if (val < 10**8):
            printf (output_file, '{}\t{:.0f}\n' .format (item['mode'], item['maxVal']))
        else:
            printf (output_file, '{}\t{}\n' .format (item['mode'], item['maxVal']))            


def RmseOfVec (vec):
    """
    given a vector of errors, calculate the RMSE
    """
    return (math.sqrt (sum([item**2 for item in vec])/len(vec)))/len(vec)

def error (str2print):
    """
    Print an error msg and exit.
    """
    print (f'Error: {str2print}')
    exit  ()

def checkIfInputFileExists (relativePathToInputFile):
    """
    Check whether an input file, given by its relative path, exists.
    If the file doesn't exist - exit with a proper error msg.
    """
    if not (os.path.isfile (relativePathToInputFile)):
        error (f'the input file {relativePathToInputFile} does not exist')

def getMachineStr ():
    if (os.getcwd().find ('itamarc')>-1): # the string 'HPC' appears in the path only in HPC runs
        return 'HPC' # indicates that this sim runs on my PC
    else:
        return 'PC' # indicates that this sim runs on an HPC       

def getTracesPath():
    """
    returns the path in which the traces files are found at this machine.
    Currently, traces files should be placed merely in the "/../traces/" subdir
    """
    return '../../traces/'

def getRelativePathToTraceFile (traceFileName):
    """
    Given a trace's file name, get the relative path to this trace file.
    The function also checks whether this trace file exists; otherwise, the run finishes with an appropriate error message.
    """
    print (f'Note: we currently assume that all traces are in directory {getTracesPath()}, and in .csv format')
    RelativePathToTraceFile = f'{getTracesPath()}Caida/{traceFileName}.csv'
    checkIfInputFileExists (RelativePathToTraceFile)
    return RelativePathToTraceFile

def extractParamsFromSettingStr (str):
    """
    given a settings string, extract from it the params it represents - e.g., cntrSize, hyperSize, expSize
    """

    splittedStr = str.split ('_')
    if len(splittedStr)<1:
        error (f'in settings.extractParamsFromSettingStr(). The input str {str} does not contain mode')
    params = {'mode'        : str.split ('_')[0]}

    splittedStr = str.split ('_n')
    if len(splittedStr)>1:
        params['cntrSize'] = int(splittedStr[1].split('_')[0]) 
    splittedStr = str.split ('_e')
    if len(splittedStr)>1:
        params['expSize'] = int(splittedStr[1].split('_')[0]) 
    splittedStr = str.split ('_m')
    if len(splittedStr)>1:
        params['mantSize'] = int(splittedStr[1].split('_')[0]) 
    splittedStr = str.split ('_h')
    if len(splittedStr)>1:
        params['hyperSize'] = int(splittedStr[1].split('_')[0]) 
    return params

def makeSymmetricVec (X):
    """
    Input: a vector X of length n.
    Output: a vector Y of length 2*n, where:
    The first n items in Y are the same as in X, but in reverse order, and inverted sign.
    The next n items in Y are the same as the n items in X.
    """
    n = len(X)
    reversed_negative_part = [-x for x in X[-1::-2]]
    Y = reversed_negative_part + [X[0]] + X[1:]
    return Y

