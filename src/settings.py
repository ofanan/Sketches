# Parameters and accessory functions
# import math, random, os, pandas as pd
import os, math, itertools, numpy as np, scipy.stats as st 
from printf import printf

SEED    = 42
INF_INT = 999999999
MAX_NUM_OF_FLOWS = 2**32

# Colors for print-out messages
STDOUT_FAIL     = '\033[91m'
STDOUT_ENDC     = '\033[0m'
# Other bcolors:
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKCYAN = '\033[96m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'
   
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
VERBOSE_PLOT            = 15

F2Pmodes  = ['F2P_sr_h1', 'F2P_sr_h2', 'F2P_lr_h1', 'F2P_lr_h2', 'F2P_si_h1', 'F2P_si_h2', 'F2P_li_h1', 'F2P_li_h2'] 
F3Pmodes  = ['F3P_sr_h1', 'F3P_sr_h2', 'F3P_sr_h3', 'F3P_lr_h1', 'F3P_lr_h2', 'F3P_lr_h3'] 
FP8modes  = ['FP_e2', 'FP_e3', 'FP_e4', 'FP_e5']
FP16modes = ['FP_e5', 'FP_e8'] #, 'FP_e10'] # 'FP_e5' is FP16. 'FP_e8' is BFloat.
FP19modes = ['FP_e5'] #, 'FP_e8' is very bad --> removing it.

VECTOR_SIZE = 1000
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
         {'cntrSize' : 12, 'cntrMaxVal' : 2096128,  'hyperSize' : 2, 'hyperMaxSize' : 3, 'seadExpSize' : 5},
         {'cntrSize' : 13, 'cntrMaxVal' : 4193280,  'hyperSize' : 2, 'hyperMaxSize' : 3, 'seadExpSize' : 5},
         {'cntrSize' : 14, 'cntrMaxVal' : 8387584,  'hyperSize' : 2, 'hyperMaxSize' : 3, 'seadExpSize' : 5},
         {'cntrSize' : 15, 'cntrMaxVal' : 16776192, 'hyperSize' : 2, 'hyperMaxSize' : 3, 'seadExpSize' : 5},
         {'cntrSize' : 16, 'cntrMaxVal' : 33553408, 'hyperSize' : 2, 'hyperMaxSize' : 3, 'seadExpSize' : 5}]

# Calculate the confidence interval of an array of values ar, given its avg. Based on 
# https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
confInterval = lambda ar, avg, conf_lvl=0.99 : st.t.interval (conf_lvl, len(ar)-1, loc=avg, scale=st.sem(ar)) if np.std(ar)>0 else [avg, avg]
   

def getDf (distStr : str) -> float:
    """
    Given the string describing a 't' distribution, return its df parameter.
    """
    return float(distStr.split('_')[1])

def modesOfCntrSize (cntrSize):
    """
    Return a list of modes to consider given the counter's size.
    """
    if cntrSize==19: 
        return F3Pmodes #$$ + F2Pmodes + ['int', 'SEAD_dyn'] + FP19modes
    elif cntrSize==16: 
        return F3Pmodes #$$ + F2Pmodes + ['int', 'SEAD_dyn'] + FP16modes
    elif cntrSize==8: 
        return F3Pmodes #$$ + F2Pmodes + ['int', 'SEAD_dyn'] + FP8modes
    else:
        error (f'In settings.modesOfCntrSize(). No hard-coded list of modes for cntrSize={cntrSize}.')
 

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

def warning (str2print):
    """
    Print an error msg and exit.
    """
    print (f'{STDOUT_FAIL}Warning: {str2print}{STDOUT_ENDC}')

def error (str2print):
    """
    Print an error msg and exit.
    """
    print (f'{STDOUT_FAIL}Error: {str2print}{STDOUT_ENDC}')
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
    # print (f'Note: we currently assume that all traces are in directory {getTracesPath()}, and in .txt format')
    RelativePathToTraceFile = f'{getTracesPath()}Caida/{traceFileName}.txt'
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
    reversed_negative_part = [-item for item in X[::-1]]
    return reversed_negative_part[:-1] + X

def first_true(iterable, default=False, pred=None):
    """
    Returns the first true value in the iterable.
    If no true value is found, returns *default*
    If *pred* is not None, returns the first item
    for which pred(item) is true.

    """
    # first_true([a,b,c], x) --> a or b or c or x
    # first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    return next(filter(pred, iterable), default)

def indexOrNone(l : list, 
                elem):
    """
    if elem is found in a list, returns the first index in which it's found.
    else, return None
    """
    try:
        idx = l.index(elem)
        return idx
    except ValueError:
        return None
    
def calcPostSimStat (
        sumSqEr         : list, # sum of the square errors, collected during the sim
        numMeausures    : int, # num of error measurements  
        statType        : str = 'normRmse', # Type of the statistic to write. May be either 'normRmse', or 'Mse'
        verbose         : list = [], # verbose level, defining the type and format of output
    ) -> dict: 
    """
    Calculate the post-sim stat - e.g., MSE/RMSE, with confidence intervals. 
    The stat is based on the sum of square errors given as input.
    Return a dict of the calculated stat.  
    """
    if statType=='Mse':
        vec  = [item/numMeausures for item in sumSqEr]
    elif statType=='normRmse': # Normalized RMSE
        Rmse = [math.sqrt (item/numMeausures) for item in sumSqEr]
        vec  = [item/numMeausures for item in Rmse]
    else:
        error (f'In settings.calcPostSimStat(). Sorry, the requested statType {statType} is not supported.')
    if (VERBOSE_LOG in verbose):
        printf (self.logFile, f'statType={statType}. Vec=')
        printarFp (self.logFile, vec)
    avg           = np.average(vec)
    confIntervalVar  = confInterval (ar=vec, avg=avg)
    # maxMinRelDiff will hold the relative difference between the largest and the smallest value.
    warningField    = False # Will be set True if the difference between the largest and the smallest value is too high.
    if avg== 0:
        maxMinRelDiff   = None
    else:
        maxMinRelDiff   = (max(vec) - min(vec))/avg
        if maxMinRelDiff>0.1:
            warningField    = True
    return {
        'Avg'           : avg,
        'Lo'            : confIntervalVar[0],
        'Hi'            : confIntervalVar[1],
        'statType'      : statType,
        'maxMinRelDiff' : maxMinRelDiff,
        'warning'       : warningField  
    }

