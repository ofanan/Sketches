# Run functions analyzing an F2P counter, e.g.: print all the possible values, or the maximal values, at a concrete configuration. 
import math, random, pickle
from printf import printf
import settings
import numpy as np

import F2P_lr, F2P_sr, F2P_li

def genCntrMaster (cntrSize, hyperSize, flavor='', verbose=[]):
    """
    return an F2P's CntrMaster belonging to the selected flavor 
    """
    if flavor=='sr':
        return F2P_sr.CntrMaster(cntrSize=cntrSize, hyperSize=hyperSize, verbose=verbose)
    elif flavor=='lr':
        return F2P_lr.CntrMaster(cntrSize=cntrSize, hyperSize=hyperSize, verbose=verbose)
    elif flavor=='li':
        return F2P_li.CntrMaster(cntrSize=cntrSize, hyperSize=hyperSize, verbose=verbose)
    else:
        settings.error (f'In F2P_master.genCntrMaster(). the requested F2P flavor {flavor} is not supported.')


def printAllCntrMaxVals (flavor='sr', hyperSizeRange=None, cntrSizeRange=[], verbose=[settings.VERBOSE_RES]):
    """
    print the maximum value a cntr reach for several "configurations" -- namely, all combinations of cntrSize and hyperSize. 
    """

    for cntrSize in cntrSizeRange:
        for hyperSize in range (1,cntrSize-2) if hyperSizeRange==None else hyperSizeRange:
            myCntrMaster = genCntrMaster (flavor=flavor, cntrSize=cntrSize, hyperSize=hyperSize)
            if myCntrMaster.isFeasible==False:
                continue
            if (settings.VERBOSE_RES in verbose):
                outputFile    = open (f'../res/cntrMaxVals.txt', 'a')
            if not(myCntrMaster.isFeasible): # This combination of cntrSize and hyperSize is infeasible
                continue
            cntrMaxVal = myCntrMaster.cntrMaxVal
            if flavor=='li':
                cntrMaxVal = int(cntrMaxVal)
            if (cntrMaxVal < 10**8):
                print (f'hyperSize={hyperSize}')
                printf (outputFile, '{} cntrMaxVal={}\n' .format (myCntrMaster.genSettingsStr(), cntrMaxVal))
            else:
                printf (outputFile, '{} cntrMaxVal={}\n' .format (myCntrMaster.genSettingsStr(), cntrMaxVal))

def printAllVals (flavor='', cntrSize=8, hyperSize=2, verbose=[]):
    """
    Loop over all the binary combinations of the given counter size. 
    For each combination, print to file the respective counter, and its value. 
    The prints are sorted in an increasing order of values.
    """
    print (f'running F2P{flavor}.printAllVals().')
    myCntrMaster = genCntrMaster (flavor=flavor, cntrSize=cntrSize, hyperSize=hyperSize, verbose=verbose)
    if myCntrMaster.isFeasible==False:
        settings.error (f'The requested configuration is not feasible.')
    listOfVals = []
    for i in range (2**cntrSize):
        cntr = np.binary_repr(i, cntrSize) 
        val = myCntrMaster.cntr2num(cntr=cntr)
        if flavor=='li':
            val = int(val)
        listOfVals.append ({'cntrVec' : cntr, 'val' : val})
    listOfVals = sorted (listOfVals, key=lambda item : item['val'])
    
    if (settings.VERBOSE_RES in verbose):
        outputFile    = open ('../res/{}.res' .format (myCntrMaster.genSettingsStr()), 'w')
        for item in listOfVals:
            printf (outputFile, '{}={}\n' .format (item['cntrVec'], item['val']))
    
    if (settings.VERBOSE_PCL in verbose):
        with open('../res/pcl_files/{}.pcl' .format (myCntrMaster.genSettingsStr()), 'wb') as pclOutputFile:
            pickle.dump(listOfVals, pclOutputFile) 

def coutConfData (cntrSize, hyperSize, flavor='', verbose=[]):
    """
    print basic configuration data about the requested flavor. 
    The conf' data includes cntrSize, hyperSize, Vmax, bias. 
    """
    genCntrMaster (flavor=flavor, cntrSize=cntrSize, hyperSize=hyperSize, verbose=[settings.VERBOSE_COUT_CONF])

# printAllVals (cntrSize=6, hyperSize=2, verbose=[settings.VERBOSE_RES, settings.VERBOSE_COUT_CONF, settings.VERBOSE_COUT_CNTRLINE], flavor='li') #, settings.VERBOSE_COUT_CNTRLINE
printAllCntrMaxVals (hyperSizeRange=None, cntrSizeRange=[6], verbose=[settings.VERBOSE_RES], flavor='lr')
# coutConfData (cntrSize=6, hyperSize=1, flavor='li')
