# Parameters and accessory functions
# import math, random, os, pandas as pd
import os, math, numpy as np, scipy.stats as st 
from printf import printf
# import commonFuncs 

VERBOSE_COUT_CNTRLINE   = 1 # print to stdout details about the concrete counter and its fields.
VERBOSE_DEBUG           = 2 # perform checks and debug operations during the run.
VERBOSE_RES             = 3 # print output to a .res file in the directory ../res
VERBOSE_DETAILED_RES    = 4
VERBOSE_PCL             = 5 # print output to a .pcl file in the directory ../res/pcl_files
VERBOSE_DETAILS         = 6 # print to stdout details about the counter
VERBOSE_NOTE            = 7 # print to stdout notes, e.g. when the target cntr value is above its max or below its min.
VERBOSE_LOG             = 8
VERBOSE_DETAILED_LOG    = 9
VERBOSE_PROGRESS        = 10 # Print periodical output notifying the progress. Used to control long runs. 
# Configurations to be run. 
# For cntrSize<8, the conf' the values are unrealistically small, and used only for checks and debugging.
# For cntrSize>=8, cntrMaxVal is calculated by that reached by F2P stat, and hyperSize is the corresponding hyper-exponent field size in F2P stat.
# hyperMaxSize is 
# expSize is the minimal needed for SEAD stat to reach the requested value.
Confs = [{'cntrSize' : 2,  'cntrMaxVal' :  16,       'hyperSize' : 1, 'hyperMaxSize' : 1, 'f2pExpSize' : 1, 'seadExpSize' : 1, 'tetraSize' : 1,'tetraMaxSize' : 1},
         {'cntrSize' : 4,  'cntrMaxVal' :  32,       'hyperSize' : 1, 'hyperMaxSize' : 1, 'f2pExpSize' : 1, 'seadExpSize' : 2, 'tetraSize' : 1,'tetraMaxSize' : 1},
         {'cntrSize' : 5,  'cntrMaxVal' :  100,      'hyperSize' : 1, 'hyperMaxSize' : 1, 'f2pExpSize' : 1, 'seadExpSize' : 2, 'tetraSize' : 1,'tetraMaxSize' : 1},
         {'cntrSize' : 6,  'cntrMaxVal' :  200,      'hyperSize' : 1, 'hyperMaxSize' : 1, 'f2pExpSize' : 1, 'seadExpSize' : 2, 'tetraSize' : 1,'tetraMaxSize' : 1},
         {'cntrSize' : 7,  'cntrMaxVal' :  300,      'hyperSize' : 1, 'hyperMaxSize' : 1, 'f2pExpSize' : 1, 'seadExpSize' : 2, 'tetraSize' : 1,'tetraMaxSize' : 1},
         {'cntrSize' : 8,  'cntrMaxVal' :  1488888,  'hyperSize' : 2, 'hyperMaxSize' : 3, 'f2pExpSize' : 3, 'seadExpSize' : 5, 'tetraSize' : 1,'tetraMaxSize' : 1},
         {'cntrSize' : 9,  'cntrMaxVal' :  2994160,  'hyperSize' : 2, 'hyperMaxSize' : 3, 'f2pExpSize' : 4, 'seadExpSize' : 5},
         {'cntrSize' : 10, 'cntrMaxVal' :  6004704,  'hyperSize' : 2, 'hyperMaxSize' : 3, 'f2pExpSize' : 4, 'seadExpSize' : 5},
         {'cntrSize' : 11, 'cntrMaxVal' :  12025792, 'hyperSize' : 2, 'hyperMaxSize' : 3, 'f2pExpSize' : 4, 'seadExpSize' : 5},
         {'cntrSize' : 12, 'cntrMaxVal' :  24067968, 'hyperSize' : 2, 'hyperMaxSize' : 3, 'f2pExpSize' : 4, 'seadExpSize' : 5},
         {'cntrSize' : 13, 'cntrMaxVal' :  48152320, 'hyperSize' : 2, 'hyperMaxSize' : 3, 'f2pExpSize' : 4, 'seadExpSize' : 5},
         {'cntrSize' : 14, 'cntrMaxVal' :  96321024, 'hyperSize' : 2, 'hyperMaxSize' : 3, 'f2pExpSize' : 4, 'seadExpSize' : 5},
         {'cntrSize' : 15, 'cntrMaxVal' : 192658432, 'hyperSize' : 2, 'hyperMaxSize' : 3, 'f2pExpSize' : 4, 'seadExpSize' : 5},
         {'cntrSize' : 16, 'cntrMaxVal' : 385333248, 'hyperSize' : 2, 'hyperMaxSize' : 3, 'f2pExpSize' : 4, 'seadExpSize' : 5}]

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
        mode = splitted_line[0]
        list_of_dict = [item['mode'] for item in list_of_dicts if item['mode']==mode]
        if (mode not in [item['mode'] for item in list_of_dicts if item['mode']==mode]):
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

def check_if_input_file_exists (relative_path_to_input_file):
    """
    Check whether an input file, given by its relative path, exists.
    If the file doesn't exist - exit with a proper error msg.
    """
    if not (os.path.isfile (relative_path_to_input_file)):
        error (f'the input file {relative_path_to_input_file} does not exist')

def getMachineStr ():
    if (os.getcwd().find ('itamarc')>-1): # the string 'HPC' appears in the path only in HPC runs
        return 'HPC' # indicates that this sim runs on my PC
    else:
        return 'PC' # indicates that this sim runs on an HPC       
