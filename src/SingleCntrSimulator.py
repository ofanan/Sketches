"""
Controller that runs single-counter simulations, using various types of counters. 
"""
import os, math, pickle, time, random #sys
from printf import printf, printar, printarFp
import numpy as np #, scipy.stats as st, pandas as pd
import settings, Cntr, CEDAR, Morris, AEE, FP, SEAD_stat, SEAD_dyn   
import F2P_sr, F2P_lr, F2P_li, F2P_li_ds, F2P_si, F3P_sr, F3P_lr, F3P_li, F3P_si   
from settings import warning, error, VERBOSE_RES, VERBOSE_PCL, VERBOSE_DETAILS, VERBOSE_COUT_CONF, VERBOSE_COUT_CNTRLINE
from settings import getFxpSettings
from datetime import datetime

class SingleCntrSimulator (object):
    """
    Controller that runs single-counter simulations, using various types of counters and configurations. 
    """

    def __init__ (self, 
                 seed           = settings.SEED,
                 verbose=[]): # defines which outputs would be written to .res / .pcl output files. See the VERBOSE macros as settings.py. 
        
        self.seed    = seed
        random.seed (settings.SEED)
        
        self.verbose = verbose
        if settings.VERBOSE_DETAILED_RES in self.verbose:
            self.verbose.append (VERBOSE_RES)
        if not (VERBOSE_PCL in self.verbose):
            print ('Note: verbose does not include .pcl')  
        
        pwdStr = os.getcwd()
        if (pwdStr.find ('itamarc')>-1): # the string 'HPC' appears in the path only in HPC runs
            self.machineStr  = 'HPC' # indicates that this sim runs on my PC
        else:
            self.machineStr  = 'PC' # indicates that this sim runs on an HPC       
        # generate directories for the output files if not exist
        if not (os.path.exists('../res')):
            os.makedirs ('../res')
        if not (os.path.exists('../res/log_files')):
            os.makedirs ('../res/log_files')
        if not (os.path.exists('../res/pcl_files')):
            os.makedirs ('../res/pcl_files')
    
    def writeProgress (self, expNum=-1, infoStr=None):
        """
        If the verbose requires that, report the progress to self.log_file
        """ 
        if not (settings.VERBOSE_PROGRESS in self.verbose):
            return
        if infoStr==None:
            printf (self.log_file, f'starting experiment{expNum}\n')
        else:
            printf (self.log_file, f'{infoStr}\n')
    
    def runSingleCntrSingleModeWrEr (self, pclOutputFile=None):
        """
        Run a single counter of mode self.mode (self.mode is the approximation cntr architecture - e.g., 'F2P', 'CEDAR').  
        Collect and write statistics about the write ("hit time") errors.
        "Hit time" error (aka "wr error") is the diff between the value the cntr represent, and
        the # of increments ("hit time") needed to make the cntr reach that value.
        For each such hit time, we calculate the relative error, defined as (cntr_val - real_val)/real_val.
        For each experiment, we calculate the avg of these relative error measurements along the simulation.
        This calculation conforms to the definition in the paper CEDAR.
        """
        self.erType                  = 'wrEr'
        self.cntrRecord[self.erType] = [0] * self.numOfExps
        self.numOfPoints             = [0] * self.numOfExps # self.numOfPoints[j] will hold the number of points collected for statistic at experiment j. The number of points varies, as it depends upon the random process of increasing the approximated cntr. 
        for expNum in range(self.numOfExps):
            realValCntr = 0 # will cnt the real values (the accurate value)
            cntrVal     = 0 # will cnt the counter's value
            self.cntrRecord['cntr'].rstCntr ()
            self.cntrRecord['sampleProb'] = 1 # probability of sampling
            self.writeProgress (expNum)
            while (cntrVal < self.maxRealVal):
                realValCntr += 1
                if (self.cntrRecord['sampleProb']==1 or random.random() < self.cntrRecord['sampleProb']): # sample w.p. self.cntrRecord['sampleProb']
                    cntrAfterInc = self.cntrRecord['cntr'].incCntr (factor=int(1), mult=False, verbose=self.verbose)
                    cntrNewVal   = cntrAfterInc['val'] / self.cntrRecord['sampleProb']
                    if (VERBOSE_DETAILS in self.verbose): 
                        print ('realVal={:.0f} oldVal={:.0f}, cntrWoScaling={:.0f}, cntrNewValScaled={:.0f}, maxRealVal={:.0f}'
                               .format (realValCntr, cntrVal, cntrAfterInc['val'], cntrNewVal, self.maxRealVal))
                    if (cntrNewVal != cntrVal): # the counter was incremented
                        cntrVal = cntrNewVal
                        self.cntrRecord['wrEr'][expNum] += abs(realValCntr - cntrVal)/realValCntr
                        self.numOfPoints       [expNum] += 1  
                    if self.dwnSmple:
                        if cntrAfterInc['val']==self.cntrRecord['cntr'].cntrMaxVal: # the cntr overflowed --> downsample
                            self.cntrRecord['cntr'].incCntr (mult=True, factor=1/2)
                            self.cntrRecord['sampleProb'] /= 2
                        if (VERBOSE_DETAILS in self.verbose): 
                            print ('smplProb={}' .format (self.cntrRecord['sampleProb'])) 
                    else:
                        if cntrAfterInc['val']==self.cntrRecord['cntr'].cntrMaxVal: # the cntr reached its maximum values and no dwon-sample is used --> finish this experiment
                            break  
 
        if (settings.VERBOSE_LOG in self.verbose):
            printf (self.log_file, f'diff vector={self.cntrRecorwrErimeVar}\n\n')

        self.cntrRecord['wrEr'] = [self.cntrRecord['wrEr'][expNum]/self.numOfPoints[expNum] for expNum in range(self.numOfExps)] 
        if (settings.VERBOSE_LOG in self.verbose):
            printf (self.log_file, 'wrEr=\n{:.3f}\n, ' .format (self.cntrRecord['wrEr']))
        
        wrErAvg             = np.average    (self.cntrRecord['wrEr'])
        wrErConfInterval = settings.confInterval (ar=self.cntrRecord['wrEr'], avg=wrErAvg)
        dict = {'erType'            : self.erType,
                'numOfExps'         : self.numOfExps,
                'mode'              : self.cntrRecord['mode'],
                'cntrSize'          : self.cntrSize, 
                'cntrMaxVal'        : self.cntrMaxVal,
                'settingStr'        : self.cntrRecord['cntr'].genSettingsStr(),
                'Avg'               : wrErAvg,
                'Lo'                : wrErConfInterval[0],
                'Hi'                : wrErConfInterval[1]}
        self.dumpDictToPcl      (dict, pclOutputFile)
        self.writeDictToResFile (dict)
    
    def dumpDictToPcl (self, dict, pclOutputFile):
        """
        Dump a single dict of data into pclOutputFile
        """
        if (VERBOSE_PCL in self.verbose):
            pickle.dump(dict, pclOutputFile) 
    
    def writeDictToResFile (self, dict):
        """
        Write a single dict of data into resOutputFile
        """
        if (VERBOSE_RES in self.verbose):
            printf (self.resFile, f'{dict}\n\n') 
    
    def runSingleCntrSingleModeWrRmse (self, pclOutputFile=None):
        """
        Run a single counter of mode self.mode (self.mode is the approximation cntr architecture - e.g., 'F2P', 'CEDAR').  
        Collect and write statistics about the write ("hit time") errors.
        "Hit time" error (aka "wr error") is the diff between the value the cntr represent, and
        the # of increments ("hit time") needed to make the cntr reach that value.
        The type of statistic collected is the Round Square Mean Error of such write errors.
        """
        self.cntrRecord['sumSqAbsEr'] = [0] * self.numOfExps # self.cntrRecord['sumSqAbsEr'][j] will hold the sum of the square absolute errors collected at experiment j. 
        self.cntrRecord['sumSqRelEr'] = [0] * self.numOfExps # self.cntrRecord['sumSqRelEr'][j] will hold the sum of the square relative errors collected at experiment j. 
        self.numOfPoints              = [0] * self.numOfExps # self.numOfPoints[j] will hold the number of points collected for statistic at experiment j. The number of points varies, as it depends upon the random process of increasing the approximated cntr. 
        for expNum in range(self.numOfExps):
            if settings.VERBOSE_LOG in self.verbose:
                printf (self.log_file, f'***exp #{expNum}***\n')
            realValCntr = 0 # will cnt the real values (the accurate value)
            cntrVal     = 0 # will cnt the counter's value
            self.cntrRecord['cntr'].rstCntr ()
            self.cntrRecord['sampleProb'] = 1 # probability of sampling
            self.writeProgress (expNum)
            while cntrVal < self.maxRealVal:
                realValCntr += 1
                if (self.cntrRecord['sampleProb']==1 or random.random() < self.cntrRecord['sampleProb']): # sample w.p. self.cntrRecord['sampleProb']
                    cntrValAfterInc = self.cntrRecord['cntr'].incCntrBy1GetVal ()
                    cntrNewVal   = cntrValAfterInc / self.cntrRecord['sampleProb']
                    if (VERBOSE_DETAILS in self.verbose): 
                        print ('realVal={:.0f} oldVal={:.0f}, cntrWoScaling={:.0f}, cntrNewValScaled={:.0f}'
                               .format (realValCntr, cntrVal, cntrValAfterInc, cntrNewVal))
                    if (cntrNewVal != cntrVal): # the counter was incremented
                        cntrVal = cntrNewVal
                        sqEr = (realValCntr - cntrVal)**2
                        self.cntrRecord['sumSqAbsEr'][expNum] += sqEr
                        self.cntrRecord['sumSqRelEr'][expNum] += sqEr/realValCntr**2
                        self.numOfPoints             [expNum] += 1
                        if settings.VERBOSE_LOG in self.verbose:
                            printf (self.log_file, 'realValCntr={}, cntrVal={}, added sumSqEr={:.4f}\n' .format (realValCntr, cntrVal, ((realValCntr - cntrVal)/realValCntr)**2))

                    if self.dwnSmple:
                        if cntrValAfterInc==self.cntrRecord['cntr'].cntrMaxVal: # the cntr overflowed --> downsample
                            self.cntrRecord['cntr'].incCntr (mult=True, factor=1/2)
                            self.cntrRecord['sampleProb'] /= 2
                        if (VERBOSE_DETAILS in self.verbose): 
                            print ('smplProb={}' .format (self.cntrRecord['sampleProb'])) 
                    else:
                        if cntrValAfterInc==self.cntrRecord['cntr'].cntrMaxVal: # the cntr reached its maximum values and no down-sample is used --> finish this experiment
                            break  
 
        for rel_abs_n in [True, False]:
            for statType in ['Mse', 'normRmse']:
                dict = self.calcPostSimStat(
                    statType    = statType, 
                    sumSqEr     = self.cntrRecord['sumSqRelEr'] if rel_abs_n else self.cntrRecord['sumSqAbsEr'],
                )
                dict['rel_abs_n']   = rel_abs_n
                dict['erType']      = 'WrRmse'
                self.dumpDictToPcl       (dict, pclOutputFile)
                self.writeDictToResFile  (dict)


    def runSingleCntrSingleModeRdRmse (self, pclOutputFile=None): 
        """
        Run a single counter of mode self.mode (self.mode is the approximation cntr architecture - e.g., 'F2P', 'CEDAR').  
        Collect and write statistics about the errors w.r.t. the real cntr (measured) value.
        The error is calculated upon each increment of the real cntr (measured) value, 
        as the difference between the measured value, and the value represented by the cntr.
        The type of statistic collected is the Round Mean Square Error of such write errors.
        """
    
        self.cntrRecord['sumSqAbsEr'] = [0] * self.numOfExps # self.cntrRecord['sumSqAbsEr'][j] will hold the sum of the square absolute errors collected at experiment j. 
        self.cntrRecord['sumSqRelEr'] = [0] * self.numOfExps # self.cntrRecord['sumSqRelEr'][j] will hold the sum of the square relative errors collected at experiment j. 
        self.numOfPoints           = [self.maxRealVal] * self.numOfExps # self.numOfPoints[j] will hold the number of points collected for statistic at experiment j. The number of points varies, as it depends upon the random process of increasing the approximated cntr. 
        for expNum in range(self.numOfExps):
            realValCntr = 0 # will cnt the real values (the accurate value)
            cntrVal     = 0 # will cnt the counter's value
            self.cntrRecord['cntr'].rstCntr ()
            self.cntrRecord['sampleProb'] = 1 # probability of sampling
            self.maxRealVal = self.cntrMaxVal if (self.maxRealVal==None) else self.maxRealVal 
            self.writeProgress (expNum)
            while realValCntr < self.maxRealVal:
                realValCntr += 1
                if (self.cntrRecord['sampleProb']==1 or random.random() < self.cntrRecord['sampleProb']): # sample w.p. self.cntrRecord['sampleProb']
                    cntrValAfterInc = self.cntrRecord['cntr'].incCntrBy1GetVal ()
                    cntrNewVal      = cntrValAfterInc / self.cntrRecord['sampleProb']
                    if (VERBOSE_DETAILS in self.verbose): 
                        if self.dwnSmple:
                            print ('realVal={:.0f} oldVal={:.0f}, cntrWoScaling={:.0f}, cntrNewValScaled={:.0f}, maxRealVal={:.0f}'
                                   .format (realValCntr, cntrVal, cntrValAfterInc, cntrNewVal, self.maxRealVal))
                        else:
                            print ('realVal={:.0f} cntrOldVal={:.0f}, cntrNewVal={:.0f}'
                                   .format (realValCntr, cntrVal, cntrValAfterInc, cntrNewVal))
                    cntrVal = cntrNewVal
                    if (self.dwnSmple and cntrAfterInc['cntrVec']==self.cntrRecord['cntr'].cntrMaxVec): # the cntr overflowed --> downsample
                        self.cntrRecord['cntr'].incCntr (mult=True, factor=1/2)
                        self.cntrRecord['sampleProb'] /= 2
                        if (VERBOSE_DETAILS in self.verbose): 
                            print ('smplProb={}' .format (self.cntrRecord['sampleProb'])) 
                sqEr = (realValCntr - cntrVal)**2
                self.cntrRecord['sumSqAbsEr'][expNum] += sqEr
                self.cntrRecord['sumSqRelEr'][expNum] += sqEr/realValCntr**2
    
        for rel_abs_n in [True, False]:
            for statType in ['Mse', 'normRmse']:
                dict = self.calcPostSimStat(
                    statType    = statType, 
                    sumSqEr     = self.cntrRecord['sumSqRelEr'] if rel_abs_n else self.cntrRecord['sumSqAbsEr'],
                )
                dict['rel_abs_n']   = rel_abs_n
                dict['erType']      = 'RdRmse'
                self.dumpDictToPcl       (dict, pclOutputFile)
                self.writeDictToResFile  (dict)
        
    def runSingleCntrSingleModeRdEr (self, pclOutputFile=None): 
        """
        Run a single counter of mode self.mode (self.mode is the approximation cntr architecture - e.g., 'F2P', 'CEDAR').  
        Collect and write statistics about the errors w.r.t. the real cntr (measured) value.
        The error is calculated upon each increment of the real cntr (measured) value, 
        as the difference between the measured value, and the value represented by the cntr.
        """
    
        self.cntrRecord['RdEr'] = [0] * self.numOfExps
        self.numOfPoints        = [self.maxRealVal] * self.numOfExps # self.numOfPoints[j] will hold the number of points collected for statistic at experiment j. The number of points varies, as it depends upon the random process of increasing the approximated cntr. 
    
        for expNum in range(self.numOfExps):
            realValCntr = 0 # will cnt the real values (the accurate value)
            cntrVal     = 0 # will cnt the counter's value
            self.cntrRecord['cntr'].rstCntr ()
            self.cntrRecord['sampleProb'] = 1 # probability of sampling
            self.writeProgress (expNum)
            while realValCntr < self.maxRealVal:
                realValCntr += 1
                if (self.cntrRecord['sampleProb']==1 or random.random() < self.cntrRecord['sampleProb']): # sample w.p. self.cntrRecord['sampleProb']
                    cntrAfterInc = self.cntrRecord['cntr'].incCntr (factor=int(1), mult=False, verbose=self.verbose)
                    cntrNewVal   = cntrAfterInc['val'] / self.cntrRecord['sampleProb']
                    if (VERBOSE_DETAILS in self.verbose): 
                        print ('realVal={:.0f} oldVal={:.0f}, cntrWoScaling={:.0f}, cntrNewValScaled={:.0f}, maxRealVal={:.0f}'
                               .format (realValCntr, cntrVal, cntrAfterInc['val'], cntrNewVal, self.maxRealVal))
                    cntrVal = cntrNewVal
                    if (self.dwnSmple and cntrAfterInc['cntrVec']==self.cntrRecord['cntr'].cntrMaxVec): # the cntr overflowed --> downsample
                        self.cntrRecord['cntr'].incCntr (mult=True, factor=1/2)
                        self.cntrRecord['sampleProb'] /= 2
                        if (VERBOSE_DETAILS in self.verbose): 
                            print ('smplProb={}' .format (self.cntrRecord['sampleProb'])) 
                self.cntrRecord['RdEr'][expNum] += abs(realValCntr - cntrVal)/realValCntr
 
        if (settings.VERBOSE_LOG in self.verbose):
            printf (self.log_file, f'diff vector={self.cntrRecorRdErimeVar}\n\n')

        self.cntrRecord['RdEr'] = [self.cntrRecord['RdEr'][expNum]/self.numOfPoints[expNum] for expNum in range(self.numOfExps)] 
        if (settings.VERBOSE_LOG in self.verbose):
            printf (self.log_file, 'RdEr=\n{:.3f}\n, ' .format (self.cntrRecord['RdEr']))
        
        rdErAvg                 = np.average    (self.cntrRecord['RdEr'])
        rdErConfInterval        = settings.confInterval (ar=self.cntrRecord['RdEr'], avg=rdErAvg)
        dict = {'erType'            : self.erType,
                'numOfExps'         : self.numOfExps,
                'mode'              : self.cntrRecord['mode'],
                'cntrSize'          : self.cntrSize, 
                'cntrMaxVal'        : self.cntrMaxVal,
                'settingStr'        : self.cntrRecord['cntr'].genSettingsStr(),
                'Avg'               : rdErAvg,
                'Lo'                : rdErConfInterval[0],
                'Hi'                : rdErConfInterval[1]}
        self.dumpDictToPcl       (dict, pclOutputFile)
        self.writeDictToResFile  (dict)
        
    def calcPostSimStat (
            self,
            sumSqEr  : list, # sum of the square errors, collected during the sim
            statType : str = 'normRmse', # Type of the statistic to write. May be either 'normRmse', or 'Mse'
            ) -> dict: 
        """
        Calculate and potentially print to .log and/or .res file (based on self.verbose) the RMSE statistics based on the values measured and stored in self.cntrRecord['sumSqEr'].
        Return a dict of the calculated data.  
        """
        if statType=='Mse':
            expResults  = [sumSqEr[expNum]/self.numOfPoints[expNum] for expNum in range(self.numOfExps)]
        elif statType=='normRmse': # Normalized RMSE
            Rmse = [math.sqrt (sumSqEr[expNum]/self.numOfPoints[expNum]) for expNum in range(self.numOfExps)]
            expResults  = [              Rmse[expNum]/self.numOfPoints[expNum]  for expNum in range(self.numOfExps)]
        else:
            error (f'In SingleCntrSimulator.calcPostSimStat(). Sorry, the requested statType {statType} is not supported.')
        if (settings.VERBOSE_LOG in self.verbose):
            printf (self.log_file, 'expResults=')
            printarFp (self.log_file, expResults)
        
        expResultsAvg          = np.average (expResults)
        expResultsConfInterval = settings.confInterval (ar=expResults, avg=expResultsAvg)
        # maxMinRelDiff will hold the relative difference between the largest and the smallest value.
        warningField    = False # Will be set True if the difference between the largest and the smallest value is too high.
        if expResultsAvg==0:
            maxMinRelDiff = None  
        else:
            maxMinRelDiff = (max(expResults) - min(expResults))/expResultsAvg
            if maxMinRelDiff>0.1:
                warningField = True
            
        return {
            'numOfExps'     : self.numOfExps,
            'mode'          : self.cntrRecord['mode'],
            'settingStr'    : self.cntrRecord['cntr'].genSettingsStr(),
            'statType'      : statType,
            'cntrSize'      : self.cntrSize, 
            'cntrMaxVal'    : self.cntrMaxVal,
            'Avg'           : expResultsAvg,
            'Lo'            : expResultsConfInterval[0],
            'Hi'            : expResultsConfInterval[1],
            'maxMinRelDiff' : maxMinRelDiff, 
            'warning'       : warningField 
        }

    def measureResolutionsByModes (
            self, 
            delPrevPcl  = False, # When True, delete the previous .pcl file, if exists
            cntrSizes   = [], 
            expSize     = None, # num of bits in the exponent to run  
            modes       = [], # modes (type of counter) to run
            ) -> None:    
        """
        Loop over all requested modes and cntrSizes, measure the relative resolution, and write the results to output files as defined by self.verbose.
        """
        if VERBOSE_PCL in self.verbose:
            pclOutputFileName = 'resolutionByModes'
            if delPrevPcl and os.path.exists(f'../res/pcl_files/{pclOutputFileName}.pcl'):
                os.remove(f'../res/pcl_files/{pclOutputFileName}.pcl')
            pclOutputFile = open(f'../res/pcl_files/{pclOutputFileName}.pcl', 'ab+')
        for self.cntrSize in cntrSizes:
            self.conf = settings.getConfByCntrSize (cntrSize=self.cntrSize)
            self.cntrMaxVal   = self.conf['cntrMaxVal'] 
            self.hyperSize    = self.conf['hyperSize'] 
            for self.mode in modes:
                self.genCntrRecord (expSize=None if self.mode.startswith('SEAD_stat') else expSize)
                listOfVals = []
                for i in range (2**self.cntrSize-2 if self.mode.startswith('SEAD_dyn') else (1 << self.cntrSize)):
                    cntrVec = np.binary_repr(i, self.cntrSize) 
                    listOfVals.append (self.cntrRecord['cntr'].cntr2num(cntrVec))           
                listOfVals = sorted (listOfVals)
                points = {'X' : listOfVals[:len(listOfVals)-1], 'Y' : [(listOfVals[i+1]-listOfVals[i])/listOfVals[i+1] for i in range (len(listOfVals)-1)]}
                if VERBOSE_PCL in self.verbose:
                    self.dumpDictToPcl ({'mode' : self.mode, 'cntrSize' : self.cntrSize, 'points' : points}, pclOutputFile)

    def measureResolutionsBySettingStrs (
            self, 
            delPrevPcl  = False, # When True, delete the previous .pcl file, if exists
            settingStrs   = [],  # Concrete settings for which the measurements will be done 
            ) -> None:    
        """
        Loop over all the desired settings, measure the relative resolution, and write the results to output files as defined by self.verbose.
        Each input setting details the cntrSize, exponent size, hyperSize, etc.
        """
        if VERBOSE_PCL in self.verbose:
            pclOutputFileName = 'resolutionBySettingStrs'
            if delPrevPcl and os.path.exists(f'../res/pcl_files/{pclOutputFileName}.pcl'):
                os.remove(f'../res/pcl_files/{pclOutputFileName}.pcl')
            pclOutputFile = open(f'../res/pcl_files/{pclOutputFileName}.pcl', 'ab+')
        for settingStr in settingStrs:
            listOfVals = []
            params = settings.extractParamsFromSettingStr (settingStr)
            self.mode       = params['mode']
            self.cntrSize   = params['cntrSize']
            if self.mode=='FP': 
                self.genCntrRecord (expSize=params['expSize'])
            else: 
                self.hyperSize   = params['hyperSize']
                self.genCntrRecord (expSize=None)        
            for i in range (2**self.cntrSize-2 if self.mode.startswith('SEAD_dyn') else (1 << self.cntrSize)):
                cntrVec = np.binary_repr(i, self.cntrSize) 
                listOfVals.append (self.cntrRecord['cntr'].cntr2num(cntrVec))           
            listOfVals = sorted (listOfVals)
            points = {'X' : listOfVals[:len(listOfVals)-1], 'Y' : [(listOfVals[i+1]-listOfVals[i])/listOfVals[i+1] for i in range (len(listOfVals)-1)]}
            if VERBOSE_PCL in self.verbose:
                self.dumpDictToPcl ({'settingStr' : settingStr, 'points' : points}, pclOutputFile)

    def genCntrRecord (self,
                       expSize=None, # When expSize==None, read the expSize from the hard-coded configurations in settings.py 
                       ):
        """
        Set self.cntrRecord, which holds the counters to run
        """

        if self.mode.startswith ('F2P') or self.mode.startswith('F3P'):
            cntrMaster = genCntrMasterFxp (
                cntrSize        = self.cntrSize, 
                numCntrs        = 1,
                fxpSettingStr   = self.mode,
                verbose         = self.verbose   
            )
            self.cntrRecord = {'mode' : self.mode, 'cntr' : cntrMaster}
        elif (self.mode=='FP'):
            if expSize==None:
                settings.error ('In SingleCntrSimulator.genCntrRecord(). For generating an FP.CntrMaster you must specify an expSize')
            self.cntrRecord = {'mode' : 'FP', 'cntr' : FP.CntrMaster(cntrSize=self.cntrSize, expSize=expSize, verbose=self.verbose)}
        elif (self.mode.startswith('SEAD_stat')):
            self.expSize      = self.conf['seadExpSize'] if expSize==None else expSize
            self.cntrRecord = {'mode' : self.mode, 'cntr' : SEAD_stat.CntrMaster(cntrSize=self.cntrSize, expSize=self.expSize, verbose=self.verbose)}
        elif (self.mode.startswith('SEAD_dyn')):
            self.cntrRecord = {'mode' : self.mode, 'cntr' : SEAD_dyn.CntrMaster(cntrSize=self.cntrSize)}
        elif (self.mode=='CEDAR'):
            self.cntrRecord = {'mode' : self.mode, 'cntr' : CEDAR.CntrMaster(cntrSize=self.cntrSize, cntrMaxVal=self.cntrMaxVal)}
        elif (self.mode=='Morris'):
            self.cntrRecord = {'mode' : self.mode, 'cntr' : Morris.CntrMaster(cntrSize=self.cntrSize, cntrMaxVal=self.cntrMaxVal)}
        else:
            settings.error ('mode {} that you chose is not supported' .format (self.mode))


    def runSingleCntrSingleMode (
        self, 
        cntrSize, 
        mode         = [], # modes (type of counter) to run. E.g., 'F2P_li_h2', 'Morris'  
        maxRealVal   = None, # The maximal value to be counted at each experiment, possibly using down-sampling. When None (default), will be equal to cntrMaxval.
        cntrMaxVal   = None, # cntrMaxVal - The maximal value that the cntr can represent w/o down-sampling. When None (default), take cntrMaxVal from settings.Confs.  global parameter (found in this file). 
        expSize      = None, # Size of the exponent. Relevant only for Static SEAD counter. If cntrMaxVal==None (default), take expSize from settings.Confs global parameter (found in this file). 
        numOfExps    = 1,    # number of experiments to run. 
        dwnSmple     = False,# When True, down-sample each time the counter's maximum value is reached.
        erTypes      = [],   # either 'RdEr', 'WrEr', 'RdRmse' or 'WrRmse'.
        rel_abs_n    = True # When True, consider rel err. Else, consider abs err.
    ):
        """
        Run a single counter for the given mode for the requested numOfExps, and write the results (statistics
        about the absolute/relative error) to a .res file.
        """        
        self.cntrSize       = cntrSize
        self.cntrMaxVal     = cntrMaxVal 
        self.expSize        = expSize
        self.numOfExps      = numOfExps
        self.dwnSmple       = dwnSmple
        self.erTypes        = erTypes # the error modes to calculate. See possible erTypes in the documentation above.
        self.mode           = mode
        if (settings.VERBOSE_DETAILED_LOG in self.verbose): # a detailed log include also all the prints of a simple log
            verbose.append(settings.VERBOSE_LOG)
        if cntrMaxVal==None:
            self.conf = settings.getConfByCntrSize (cntrSize=self.cntrSize)
            self.cntrMaxVal   = self.conf['cntrMaxVal']
            self.hyperSize    = self.conf['hyperSize'] 
            self.hyperMaxSize = self.conf['hyperMaxSize'] 
        else:
            self.cntrMaxVal   = cntrMaxVal 
            if mode.startswith ('F2P') or mode.startswith ('F3P'):
                numSettings     = getFxpSettings (mode)
                self.nSystem    = numSettings['nSystem']
                self.flavor     = numSettings['flavor']
                self.hyperSize  = numSettings['hyperSize']

        self.genCntrRecord () # Set self.cntrRecord, which holds the counter to run
        self.maxRealVal         = self.cntrMaxVal 
        if self.cntrRecord['cntr'].cntrMaxVal < self.maxRealVal and (not(self.dwnSmple)):
            warning ('The counter of type {}, cntrSize={}, hyperSize={}, can reach max val={} which is smaller than the requested maxRealVal {}, and no dwn smpling was used' . format (self.cntrRecord['mode'], self.cntrSize, self.hyperSize, self.cntrRecord['cntr'].cntrMaxVal, self.maxRealVal))

        # open output files
        outputFileStr = '1cntr_{}{}' .format (self.machineStr, '_w_dwnSmpl' if self.dwnSmple else '')
        if (VERBOSE_RES in self.verbose):
            self.resFile = open (f'../res/{outputFileStr}.res', 'a+')
            
        print ('Started running runSingleCntr at t={}. erTypes={} mode={}, cntrSize={}, maxRealVal={}, cntrMaxVal={}' .format (
                datetime.now().strftime("%H:%M:%S"), self.erTypes, self.mode, self.cntrSize, self.maxRealVal, self.cntrRecord['cntr'].cntrMaxVal))
        
        # run the simulation          
        for self.erType in self.erTypes:
            self.calc_MSE = False # By default, do not calculate the MSE
            if self.erType=='WrMse':
                self.calc_MSE = True
                self.erType = 'WrRmse'
            elif self.erType=='RdMse':
                self.calc_MSE = True
                self.erType = 'RdRmse'
            if not (self.erType in ['WrEr', 'WrRmse', 'RdEr', 'RdRmse', 'WrMse', 'RdMse']):
                settings.error ('Sorry, the requested error mode {self.erType} is not supported')
            pclOutputFile = None # default value
            if VERBOSE_PCL in self.verbose:
                pclOutputFile = self.openPclOuputFile (pclOutputFileName=f'{outputFileStr}.pcl')
            simT = time.time()
            infoStr = '{}_{}' .format (self.cntrRecord['cntr'].genSettingsStr(), self.erType)
            if (settings.VERBOSE_LOG in self.verbose or settings.VERBOSE_PROGRESS in self.verbose):
                self.log_file = open (f'../res/log_files/{infoStr}.log', 'w')
            self.writeProgress (infoStr=infoStr)
            getattr (self, f'runSingleCntrSingleMode{self.erType}') (pclOutputFile) # Call the corresponding function, according to erType (read/write error, regular/RMSE).
            self.closePclOuputFile(pclOutputFile)
            print ('finished. Elapsed time={:.2f} secs' .format (time.time() - simT))

    def closePclOuputFile (self, pclOutputFile):
        """
        If VERBOSE_PCL is set, close sel.fpclOutputFile
        """
        if VERBOSE_PCL in self.verbose:
            pclOutputFile.close ()

    def openPclOuputFile (self, pclOutputFileName):
        """
        If VERBOSE_PCL is set, return an pclOutputFile with the requested file name.
        Else, return None
        """
        if VERBOSE_PCL in self.verbose:
            pclOutputFile = open(f'../res/pcl_files/{pclOutputFileName}', 'ab+')
        else:
            pclOutputFile = None
        return pclOutputFile 
        
    def runSingleCntr (self, 
                       cntrSize, 
                       modes        = [], # modes (type of counter) to run  
                       maxRealVal   = None, # The maximal value to be counted at each experiment, possibly using down-sampling. When None (default), will be equal to cntrMaxval. 
                       cntrMaxVal   = None, # cntrMaxVal - The maximal value that the cntr can represent w/o down-sampling. When None (default), take cntrMaxVal from settings.Confs.  global parameter (found in this file). 
                       hyperSize    = None, # Relevant only for F2P counter. When cntrMaxVal==None (default), take hyperSize from settings.Confs global parameter (found in this file). 
                       hyperMaxSize = None, # Relevant only for F3P counter. When cntrMaxVal==None (default), take hyperMaxSize from settings.Confs global parameter (found in this file). 
                       expSize      = None, # Size of the exponent. Relevant only for Static SEAD counter. If cntrMaxVal==None (default), take expSize from settings.Confs global parameter (found in this file). 
                       numOfExps    = 1,    # number of experiments to run. 
                       dwnSmple     = False,# When True, down-sample each time the counter's maximum value is reached.
                       erTypes      = [],   # either 'RdEr', 'WrEr', 'RdRmse' or 'WrRmse'.
                       rel_abs_n    = True, # When True, consider rel err. Else, consider abs err.
                       ):
        """
        run a single counter of each given mode for the requested numOfExps.
        Write the results (statistics) as determined by self.verbose to either .pcl / .res output file.
        about the absolute/relative error) to a .res file.
        The wr ("hitting time") error  of a counter for a given is the number of increments until the counter reaches this value.
            For instance, suppose we have 100 experiments, and the number of increments until the counter's value is 10 are: 8,9,11,13. 
            Then, CEDAR-style relative error is stdev([8,9,11,13])/avg([8,9,11,13]). 
            In practice, it's fair to assume that the avg hitting time is the relevant counter's value. E.g., in the example above,
            assume that avg([8,9,11,13]) = 10.
        The read error of a counter is caluclated as follows:
            Upon each increment of the real cntr (measured) value, define the error as the difference between the measured value, 
            and the value represented by the cntr.
        """
        self.cntrSize       = cntrSize
        self.maxRealVal     = maxRealVal
        self.cntrMaxVal     = cntrMaxVal 
        self.hyperSize      = hyperSize
        self.hyperMaxSize   = hyperMaxSize
        self.expSize        = expSize
        self.numOfExps      = numOfExps
        self.dwnSmple       = dwnSmple
        self.erTypes        = erTypes # the error modes to calculate. See possible erTypes in the documentation above.
        if (settings.VERBOSE_DETAILED_LOG in self.verbose): # a detailed log include also all the prints of a simple log
            verbose.append(settings.VERBOSE_LOG)
        for self.mode in modes:
            self.runSingleCntrSingleMode ()
        return

def genCntrMasterFxp (
        cntrSize        : int, 
        numCntrs        : int = 1,
        fxpSettingStr   : str  = None,
        hyperSize       : int  = None, 
        nSystem         : str  = None, # 'F2P' or 'F3P
        flavor          : str  = None, # 'sr' / 'lr' / 'si' / 'li' 
        # exitError       : bool = True, # When true, if the given are infeasible, exist with error.
        verbose         : list = []    # list of verboses taken from the veroses, defined in settings.py 
    ):
    """
    return a CntrMaster belonging to the selected flavor ('sr', 'lr', etc.) and number system ('F2P' or 'F3P').
    If fxpSettingStr==None, the settings are read from the other inputs.
    Else, the settings are read from fxpSettingStr.
    """
    if fxpSettingStr!=None:
        cntrSettings = getFxpSettings (fxpSettingStr)
        nSystem     = cntrSettings['nSystem']
        hyperSize   = cntrSettings['hyperSize']
        flavor      = cntrSettings['flavor']
        dwnSmpl     = cntrSettings['downSmpl']
    if nSystem=='F2P':
        if flavor=='sr':
            return F2P_sr.CntrMaster(cntrSize=cntrSize, numCntrs=numCntrs, hyperSize=hyperSize, verbose=verbose)
        elif flavor=='lr':
            return F2P_lr.CntrMaster(cntrSize=cntrSize, numCntrs=numCntrs, hyperSize=hyperSize, verbose=verbose)
        elif flavor=='li':
            if dwnSmpl:
                return F2P_li_ds.CntrMaster(cntrSize=cntrSize, numCntrs=numCntrs, hyperSize=hyperSize, verbose=verbose)
            else:
                return F2P_li.CntrMaster(cntrSize=cntrSize, numCntrs=numCntrs, hyperSize=hyperSize, verbose=verbose)
        elif flavor=='si':
            return F2P_si.CntrMaster(cntrSize=cntrSize, numCntrs=numCntrs, hyperSize=hyperSize, verbose=verbose)
        else:
            settings.error (f'In SingleCntrSimulator.genCntrMasterFxp(). the requested F2P flavor {flavor} is not supported.')

    elif nSystem=='F3P':

        if flavor=='sr':
            return F3P_sr.CntrMaster(cntrSize=cntrSize, numCntrs=numCntrs, hyperMaxSize=hyperSize, verbose=verbose)
        elif flavor=='lr':
            return F3P_lr.CntrMaster(cntrSize=cntrSize, numCntrs=numCntrs, hyperMaxSize=hyperSize, verbose=verbose)
        elif flavor=='li':
            return F3P_li.CntrMaster(cntrSize=cntrSize, numCntrs=numCntrs, hyperMaxSize=hyperSize, verbose=verbose)
        elif flavor=='si':
            return F3P_si.CntrMaster(cntrSize=cntrSize, numCntrs=numCntrs, hyperMaxSize=hyperSize, verbose=verbose)
        else:
            error (f'In SingleCntrSimulator.genCntrMasterFxp(). the requested F3P flavor {flavor} is not supported.')

    else:
        error (f'In SingleCntrSimulator.genCntrMasterFxp(). the requested number system {nSysem } is not supported.')


def getAllValsFP (cntrSize  = 8, # of bits in the cntr (WITHOUT the sign bit) 
                  expSize   = 1, # number of bits in the exp.
                  signed    = False, 
                  verbose   = [] # verbose level. See settings.py
                  ):
    """
    Loop over all the binary combinations of the given counter size.
    For each combination, get the respective counter.
    Sort by an increasing value.
    Output is according to the verbose, as defined in settings.py. In particular: 
    If the verbose include VERBOSE_RES, print to an output file the list of cntrVecs and respective values. 
    Return the (sorted) list of values.
    """
    if signed:
        cntrSize -= 1
    listOfVals = []
    myCntrMaster = FP.CntrMaster(cntrSize=cntrSize, expSize=expSize, verbose=verbose, signed=False) # For efficiency, we generate an unsigned vec; later, we will take its mirror for the negative part.
    for num in range(2 ** cntrSize):
        cntr = np.binary_repr(num, cntrSize)
        val = myCntrMaster.cntr2num(cntr)
        listOfVals.append ({'cntrVec' : cntr, 'val' : val})
    listOfVals = sorted (listOfVals, key=lambda item : item['val'])
    if VERBOSE_RES in verbose:
        outputFile = open('../res/{}.res'.format(myCntrMaster.genSettingsStr()), 'w')
        printf (outputFile, f'// bias={myCntrMaster.bias}\n')
        for item in listOfVals:
            printf(outputFile, '{}={}\n'.format(item['cntrVec'], item['val']))

    listOfVals = [item['val'] for item in listOfVals]    
    if signed:
        listOfVals = settings.makeSymmetricVec (listOfVals)
        
    return listOfVals


def getAllValsFxp (flavor='', 
   cntrSize     = 8, # size of the counter, WITHOUT the sign bit (if exists).  
   hyperSize    = 2, # size of the hyper-exp field (in F2P); Max size of the hyper-exp field (in F2P).
   nSystem      = 'F2P', # either 'F2P' or 'F3P'
   verbose      = [], #verbose level. See settings.py for details.
   signed       = False # When True, assume an additional bit for the  
):
    """
    Loop over all the binary combinations of the given counter size. 
    For each combination, get the respective counter.
    Sort by an increasing value.
    Output is according to the verbose, as defined in settings.py. In particular: 
    If the verbose include VERBOSE_RES, print to an output file the list of cntrVecs and respective values. 
    Return the (sorted) list of values.
    """
    if signed: 
        cntrSize -= 1 
    myCntrMaster = genCntrMasterFxp (nSystem=nSystem, flavor=flavor, cntrSize=cntrSize, hyperSize=hyperSize, verbose=verbose)
    if myCntrMaster.isFeasible==False:
        settings.error (f'The requested configuration is not feasible.')
    listOfVals = []
    for i in range (2**cntrSize):
        cntr = np.binary_repr(i, cntrSize) 
        val = myCntrMaster.cntr2num(cntr=cntr)
        if flavor in ['si', 'li']:
            val = int(val)
        listOfVals.append ({'cntrVec' : cntr, 'val' : val})
    listOfVals = sorted (listOfVals, key=lambda item : item['val'])
    
    if (VERBOSE_RES in verbose):
        outputFile    = open ('../res/{}.res' .format (myCntrMaster.genSettingsStr()), 'w')
        for item in listOfVals:
            printf (outputFile, '{}={}\n' .format (item['cntrVec'], item['val']))
    
    if (VERBOSE_PCL in verbose):
        with open('../res/pcl_files/{}.pcl' .format (myCntrMaster.genSettingsStr()), 'wb') as pclOutputFile:
            pickle.dump(listOfVals, pclOutputFile)

    listOfVals = [item['val'] for item in listOfVals]    
    if signed:
        listOfVals = settings.makeSymmetricVec (listOfVals)
        
    return listOfVals

def getCntrsMaxValsFxp (
        nSystem         : str, # either 'F2p' or 'F3P.
        flavor          : str  = 'sr', 
        hyperSizeRange  : list = [], # list of hyper-sizes to consider  
        cntrSizeRange   : list = [], # list of cntrSizes to consider
        # exitError       : bool = True, # When true, if the given are infeasible, exist with error.
        verbose         : list =[VERBOSE_RES]
        ):
    """
    Get the maximum value a cntr reach for several "configurations" -- namely, all combinations of cntrSize and hyperSize.
    Print the result if was requested by the VERBOSE parameter.
    Returns the cntrMaxVal of the last conf' it was called with  
    """

    for cntrSize in cntrSizeRange:
        for hyperSize in range (1,cntrSize-2) if hyperSizeRange==None else hyperSizeRange:
            myCntrMaster = genCntrMasterFxp (nSystem=nSystem, flavor=flavor, cntrSize=cntrSize, hyperSize=hyperSize)
            if not(myCntrMaster.isFeasible):
                continue
            if VERBOSE_RES in verbose:
                outputFile    = open (f'../res/cntrMaxVals.txt', 'a')
            if not(myCntrMaster.isFeasible): # This combination of cntrSize and hyperSize is infeasible
                continue
            cntrMaxVal = myCntrMaster.cntrMaxVal
            if flavor in ['si', 'li']:
                cntrMaxVal = int(cntrMaxVal)
            if (VERBOSE_RES not in verbose):
                continue
            if (cntrMaxVal < 10**8):
                printf (outputFile, '{} cntrMaxVal={}\n' .format (myCntrMaster.genSettingsStr(), cntrMaxVal))
            else:
                printf (outputFile, '{} cntrMaxVal={}\n' .format (myCntrMaster.genSettingsStr(), cntrMaxVal))
    return cntrMaxVal

def getFxpCntrMaxVal (
        cntrSize : int,
        fxpSettingStr : str
        ) -> float:
    """
    Given a string detailing the settings an F2P/F3P counter, returns its maximum representable value. 
    """
    if not(fxpSettingStr.startswith('F2P')) and not(fxpSettingStr.startswith('F3P')):
        error (f'SingleCntrSimulator.getFxpCntrMaxVal() was called with Fxp settings str={fxpSettingStr}')
    myCntrMaster = genCntrMasterFxp (cntrSize=cntrSize, fxpSettingStr=fxpSettingStr)
    return myCntrMaster.getCntrMaxVal ()


def getAllCntrsMaxValsFxP ():
    for nSystem in ['F2P']: #['F2P_si_h2', 'F2P_li_h2', 'F2P_si_h3', 'F3P_li_h3', 'F3P_si_h2', 'F3P_li_h2', 'F3P_si_h3', 'F3P_li_h3']:
        for flavor in ['si', 'li']:
            getCntrsMaxValsFxp (
                nSystem         = nSystem,  
                flavor          = flavor, 
                hyperSizeRange  = [1,2], # list of hyper-sizes to consider  
                cntrSizeRange   = range (8, 17), # list of cntrSizes to consider
                verbose         = [VERBOSE_PCL]
            )

def printAllValsFxp ():
    getAllValsFxp (
       nSystem      = 'F3P', # either 'F2P' or 'F3P'
       cntrSize     = 7, # size of the counter, WITHOUT the sign bit (if exists).  
       hyperSize    = 3, # Max size of the hyper-exp field. Relevant only for F3P. 
       flavor       = 'li', 
       verbose      = [VERBOSE_RES, VERBOSE_COUT_CONF], #verbose level. See settings.py for details.
       signed       = False # When True, assume an additional bit for the  
    )

def main ():
    # getAllCntrsMaxValsFxP ()
    maxValBy = 'F2P_li_h2'
    modes = [maxValBy, 'CEDAR', 'Morris', 'SEAD_dyn']
    for cntrSize in [16]:
        simController = SingleCntrSimulator (verbose = [VERBOSE_RES, VERBOSE_PCL])
        numSettings = getFxpSettings (maxValBy)
        cntrMaxVal = getCntrsMaxValsFxp (
            nSystem     = numSettings['nSystem'], # either 'F2p' or 'F3P.
            flavor      = numSettings['flavor'], 
            hyperSizeRange = [numSettings['hyperSize']], # list of hyper-sizes to consider  
            cntrSizeRange  = [cntrSize], # list of cntrSizes to consider
            verbose        = [],
        )
        for mode in modes:
            simController.runSingleCntrSingleMode \
                (dwnSmple       = False,  
                mode            = mode, 
                cntrSize        = cntrSize, 
                cntrMaxVal      = cntrMaxVal,
                numOfExps       = 100,
                erTypes         = ['RdRmse'], # The error modes to gather during the simulation. Options are: 'WrEr', 'WrRmse', 'RdEr', 'RdRmse' 
            )
        

        # simController = SingleCntrSimulator (verbose = [VERBOSE_RES, VERBOSE_PCL]) #VERBOSE_RES, VERBOSE_PCL],)
        # simController.measureResolutionsByModes (
        #     cntrSizes   = [8], 
        #     expSize     = 2, 
        #     modes       = ['SEAD stat', 'Morris', 'F2P_li'],
        #     delPrevPcl  = True
        #     ) 
        # simController.measureResolutionsBySettingStrs (
        #     delPrevPcl  = False, # When True, delete the previous .pcl file, if exists
        #     settingStrs = ['FP_n15_m10_e5'], # 'FP_n15_m2_e13'],  # 'FP_n7_m5_e2'# Concrete settings for which the measurements will be done 
        #     # settingStrs = ['F2Plr_n15_h2', 'F2Psr_n15_h2'],  # 'FP_n7_m5_e2'# Concrete settings for which the measurements will be done 
        #     # settingStrs = ['F2Plr_n7_h1', 'F2Psr_n7_h1'],  # 'FP_n7_m5_e2'# Concrete settings for which the measurements will be done 
        #     )    
    

if __name__ == '__main__':
    try:
        main ()
    except KeyboardInterrupt:
        print('Keyboard interrupt.')
        exit ()
        
