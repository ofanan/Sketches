# Parameters and accessory functions
# import math, random, os, pandas as pd
import os, math, itertools, numpy as np, scipy.stats as st 
from printf import printf, printarFp
np.set_printoptions(precision=1)

SEED    = 123456789012345678901234567890123456789
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
VERBOSE_RES             = 3 # print output to a .res file in the directory ../res
VERBOSE_DETAILED_RES    = 4
VERBOSE_FULL_RES        = 5
VERBOSE_PCL             = 6 # print output to a .pcl file in the directory ../res/pcl_files
VERBOSE_DETAILS         = 7 # print to stdout details about the counter
VERBOSE_NOTE            = 8 # print to stdout notes, e.g. when the target cntr value is above its max or below its min.
VERBOSE_LOG             = 10
VERBOSE_LOG_SHORT       = 11
VERBOSE_LOG_END_SIM     = 12
VERBOSE_DETAILED_LOG    = 13
VERBOSE_LOG_CNTRLINE    = 15  
VERBOSE_LOG_TRACESTAT   = 16  
VERBOSE_LOG_DWN_SMPL    = 20
VERBOSE_LOG_DWN_SMPL_D  = 21 # Detailed dwn-smpl logging
VERBOSE_PROGRESS        = 30 # Print periodical output notifying the progress. Used to control long runs.
VERBOSE_PLOT            = 40
VERBOSE_DEBUG           = 60 # perform checks and debug operations during the run.

KB = 2**10 # Kilo-Byte

F2Pmodes  = ['F2P_sr_h1', 'F2P_sr_h2', 'F2P_lr_h1', 'F2P_lr_h2', 'F2P_si_h1', 'F2P_si_h2', 'F2P_li_h1', 'F2P_li_h2'] 
F3Pmodes  = ['F3P_sr_h1', 'F3P_sr_h2', 'F3P_sr_h3', 'F3P_lr_h1', 'F3P_lr_h2', 'F3P_lr_h3'] 
FP8modes  = ['FP_e2', 'FP_e3', 'FP_e4', 'FP_e5']
FP16modes = ['FP_e5', 'FP_e8'] #, 'FP_e10'] # 'FP_e5' is FP16. 'FP_e8' is BFloat.
FP19modes = ['FP_e5'] #, 'FP_e8' is very bad --> removing it.

traceInfo = [
             {'traceName' : 'Caida1', 'traceFullName' : 'Caida1_equinix-nyc.dirA.20181220-130000.UTC.anon', 'len' : 25000000, 'numFlows' :  1801150, 'maxFlowSize' : 455156},               
             {'traceName' : 'Caida2', 'traceFullName' : 'Caida2_equinix-chicago.dirA.20160406-130000.UTC.anon', 'len' : 25000000, 'numFlows' : 852129, 'maxFlowSize' :  3128382}, 
            ]

VECTOR_SIZE = 1000
FLOW_TYPE = 'uint32'
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
confInterval = lambda ar, avg, confLvl=0.95 : st.t.interval (confLvl, len(ar)-1, loc=avg, scale=st.sem(ar)) if np.std(ar)>0 else [avg, avg]

def getMaxValByStr (maxValBy : str 
                    ):
    """
    Given maxValBy, which is an identifier of the counter by which the maximum value was calculated, return a identifyin string, to be used for naming the corresponding file/variable.
    """
    if maxValBy==None:
        return 'None'          
    return maxValBy.split('_')[0]          
    
   
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
        return F3Pmodes + F2Pmodes + ['int', 'SEAD_dyn'] + FP19modes
    elif cntrSize==16: 
        return F3Pmodes + F2Pmodes + ['int', 'SEAD_dyn'] + FP16modes
    elif cntrSize==8: 
        return F3Pmodes + F2Pmodes + ['int', 'SEAD_dyn'] + FP8modes
    else:
        error (f'In settings.modesOfCntrSize(). No hard-coded list of modes for cntrSize={cntrSize}.')
 

def getConfByCntrSize (cntrSize):
    """
    given the counter's size, return the configuration with that counter size.
    If the number of configurations with that counter's size is not 1, exit with a proper error message.
    """
    listOfConfs = [item for item in Confs if item['cntrSize']==cntrSize]
    if (len(listOfConfs)<1): 
        error (f'Sorry. No known configuration for cntrSize={cntrSize}')
    elif (len(listOfConfs)>1):
        error (f'Sorry. Too many known configurations for cntrSize={cntrSize}')
    return listOfConfs[0]
   
def getConfByCntrMaxVal (cntrSize, cntrMaxVal):
    """
    return the hard-coded configuration that has the required counter size and counter max size. 
    If the number of configurations with that counter's size is not 1, exit with a proper error message.
    """
    listOfConfs = [item for item in Confs if item['cntrSize']==cntrSize and item['cntrMaxVal']==cntrMaxVal]
    if (len(listOfConfs)<1): 
        error (f'Sorry. No known configuration for cntrSize={cntrSize} and cntrMaxVal={cntrMaxVal}')
    elif (len(listOfConfs)>1):
        error (f'Sorry. Too many known configurations for cntrSize={cntrSize} and cntrMaxVal={cntrMaxVal}')
    return listOfConfs[0]


# def getSeadExpSizeByCntrMaxVal (
#         cntrSize    : int,
#         cntrMaxVla  : int, 
#     ) -> int:
#     """
#     return the size of the exponent field in Sead_stat with the required counter size and counter max size. 
#     """
#     seadStatConfs = [
#         {'cntrSize: 8,    'cntrMaxVal: 8032,     'expSize' : 3},
#         {'cntrSize: 8,    'cntrMaxVal: 1015792,  'expSize' : 4},
#         {'cntrSize: 9,    'cntrMaxVal, 16192,    'expSize' : 3},
#         {'cntrSize: 9,    'cntrMaxVal, 2064352   'expSize' : 4},
#         {'cntrSize: ,     'cntrMaxVal,           'expSize' : },
#         {'cntrSize: ,     'cntrMaxVal,           'expSize' : },
#         {'cntrSize: ,     'cntrMaxVal,           'expSize' : },
#         {'cntrSize: ,     'cntrMaxVal,           'expSize' : },
#     ] 
    
   
def getTraceLen (
        traceName
    ):
    """
    given the trace's name, return its len (# of incs).
    """
    listOfTraces = [item for item in traceInfo if item['traceName']==traceName]
    if (len(listOfTraces)<1): 
        error (f'In settings.getTraceLen(). Sorry. No known traceInfo for traceName={traceName}')
    elif (len(listOfTraces)>1):
        error (f'In settings.getTraceLen(). Sorry. Too many known traces for trace={traceName}')
    return listOfTraces[0]['len']
   
def getTraceFullName (
        traceName
    ):
    """
    given the trace's name, return its full name, detailing its origin.
    """
    listOfTraces = [item for item in traceInfo if item['traceName']==traceName]
    if (len(listOfTraces)<1): 
        error (f'In settings.getTraceLen(). Sorry. No known traceInfo for traceName={traceName}')
    elif (len(listOfTraces)>1):
        error (f'In settings.getTraceLen(). Sorry. Too many known traces for trace={traceName}')
    return listOfTraces[0]['traceFullName']
   
def getNumFlowsByTraceName (
        traceName : str
    ):
    """
    given the trace's name, return its len (# of incs).
    """
    listOfTraces = [item for item in traceInfo if item['traceName']==traceName]
    if (len(listOfTraces)<1): 
        error (f'In settings.getNumFlowsByTraceName(). Sorry. No known traceInfo for trace={traceName}')
    elif (len(listOfTraces)>1):
        error (f'In settings.getNumFlowsByTraceName(). Sorry. Too many known traces for trace={traceName}')
    return listOfTraces[0]['numFlows']
   
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

def checkIfInputFileExists (
        relativePathToInputFile : str,
        exitError               = True
        ):
    """
    Check whether an input file, given by its relative path, exists.
    If the file exists, return True. 
    Else, print proper error msg, and:
        if exitError=True, exit with error; else, return False 
    """
    if os.path.isfile (relativePathToInputFile):
        return True
    if exitError:
        error (f'the input file {relativePathToInputFile} does not exist')
    else:
        warning (f'the input file {relativePathToInputFile} does not exist')
        return False

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

def getRelativePathToTraceFile (
        traceFileName,
        ):
    """
    Given a trace's file name, get the relative path to this trace file.
    The function doesn't checks whether this trace file exists. 
    """
    return f'{getTracesPath()}Caida/{traceFileName}'

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
        sumSqEr       : np.array, # sum of the square errors, collected at each experiment during the sim
        numMeausures  : np.array, # numMeausures[i] should hold the num of error measurements in the i-th experiment  
        statType      : str = 'normRmse', # Type of the statistic to write. May be either 'normRmse', or 'Mse'
        confLvl       : float = 0.95, # required conf' level
        verbose       : list = [], # verbose level, defining the type and format of output
        logFile       = None,
    ) -> dict: 
    """
    Calculate the post-sim stat - e.g., MSE/RMSE, with confidence intervals. 
    The stat is based on the sum of square errors given as input.
    Return a dict of the calculated stat.  
    """
    sumSqEr = np.divide (sumSqEr, numMeausures) 
    if statType=='Mse': # Normalized RMSE
        pass 
    elif statType=='normRmse': # Normalized RMSE
        sumSqEr  = np.divide (np.sqrt (sumSqEr), numMeausures) 
    else:
        error (f'In settings.calcPostSimStat(). Sorry, the requested statType {statType} is not supported.')
    if (VERBOSE_LOG in verbose):
        if logFile==None:
            error ('settings.calcPostSimStat() was called with VERBOSE_LOG, but logFile==None')
        printf (logFile, f'statType={statType}. Vec=')
        printarFp (logFile, sumSqEr)
    avg           = np.average(sumSqEr)
    confIntervalVar  = confInterval (ar=sumSqEr, avg=avg, confLvl=confLvl)
    # maxMinRelDiff will hold the relative difference between the largest and the smallest value.
    warningField    = False # Will be set True if the difference between the largest and the smallest value is too high.
    if avg== 0:
        maxMinRelDiff   = None
    else:
        maxMinRelDiff   = (np.max(sumSqEr) - np.min(sumSqEr))/avg
        if maxMinRelDiff>0.1:
            warningField    = True
    return {
        'Avg'           : avg,
        'Lo'            : confIntervalVar[0],
        'Hi'            : confIntervalVar[1],
        'statType'      : statType,
        'maxMinRelDiff' : maxMinRelDiff,
        'warning'       : warningField,
        'confLvl'       : confLvl  
    }

def getFxpSettings (mode : str) -> dict:
    """
    given the mode string of an F2P or F3P counter, get a dictionary detailing its settings (flavor and hyperExp size).
    """
    nSystem   = mode.split('_')[0]
    if (not(mode.startswith('F2P')) and not(mode.startswith('F3P'))) or len(mode.split('_h'))==1:
        error (f'In settings.getFxpSettings(). Could not get the Fxp settings of mode {mode}')
    return {
        'nSystem'   : nSystem,
        'flavor'    : mode.split(f'{nSystem}_')[1].split('_')[0],
        'hyperSize' : int(mode.split('_h')[1].split('_')[0]),
        'downSmpl'  : True if mode.endswith('_ds') else False
    }
    

def getSeadStatExpSize (
        mode : str
    )  -> int:
    """
    given the mode string of a static sead counter, get a dictionary detailing its settings (exponent).
    """
    if not(mode.startswith('SEAD_stat')):
        error (f'In settings.getStatSeadExpSize(). Could not get SEAD_stat settings of mode {mode}')
    if len (mode.split('_e'))<2:
        error (f'settings.getSeadStatExpSize() was called with wrong mode {mode}')
    return int(mode.split('_e')[1])
    
def genElapsedTimeStr (elapsedTime : float,
                       printInitStr = True, # when true, print the words 'elapsed time: '
                       ) -> str:
    """
    Returns a string that describes the elapsed time in hours, minutes, and seconds
    """
    initStr = 'elapsed time: ' if printInitStr else ''  
    if elapsedTime >= 3600:
        hours, rem = divmod(elapsedTime, 3600)
        minutes, seconds = divmod(rem, 60)
        return f'{initStr}{int(hours)}h {int(minutes)}m {seconds:.2f}s'
    elif elapsedTime >= 60:
        minutes, seconds = divmod(elapsedTime, 60)
        return f'{initStr}{int(minutes)}m {seconds:.0f}s'
    else:
        return f'{initStr}{elapsedTime:.4f} seconds'
    
def writeVecStatToFile (
        statFile,
        vec,
        str,
        numBins = 100,
    ):
    """
    Calculate and write the statistics (mean, min, max, std, binning) of a given vector to the given output file.
    """
    vec = np.array(vec)
    printf (statFile, f'// vec={str}\n')
    lenVec = int(len(vec))
    maxVec = int(np.max(vec))
    minVec = np.min(vec)
    printf (statFile, '// len(vec)={:.0f}, minVec={:.1f},  maxVec={:.1f}, avgVec={:.1f}, stdevVec={:.1f}\n' .format
           (lenVec, minVec, maxVec, np.mean(vec), np.std(vec))) 
    
    if lenVec<11: # No need to print binning data for up to 10 bins: one can merely print the data itself.
        return
    
    numBins = np.min ([numBins, int(maxVec)+1, lenVec])
    binSize = maxVec // (numBins-1)
    binVal  = np.zeros (numBins, dtype=FLOW_TYPE) 
    flowsizeOverBinsize = np.divide (vec, binSize).astype (FLOW_TYPE)
    bins = (binSize*np.arange(numBins)).astype (FLOW_TYPE) 
    for bin in range(numBins):
        binVal[bin] = np.sum(np.where(flowsizeOverBinsize==bin, 1, 0))
    printf (statFile, f'// bins:\n')
    for bin in range(numBins):
        printf (statFile, f'bin={bins[bin]}, binVal={binVal[bin]}\n')
    if minVec<0:
        warning ('in settings.writeVecStatToFile(). The minimal value in vector {} is {:.2f}' .format
                 (str, minVec))
    