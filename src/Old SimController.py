"""
Controller that runs simulations, using various types of couters. 
"""
#import itertools
# from   pathlib import Path
# from builtins import True False
from statistics import mean 
import os, math, pickle, time, random #sys
from printf import printf, printar, printarFp
import numpy as np #, scipy.stats as st, pandas as pd
import settings, SEAD, CEDAR, Morris, AEE
from datetime import datetime

def main ():
    simController = SimController (verbose = [settings.VERBOSE_PCL]) #settings.VERBOSE_RES, settings.VERBOSE_PCL],)
    # simController.measureResolutions (cntrSizes=[8, 12, 16], modes=['CEDAR', 'F2P', 'F3P', 'SEAD stat', 'SEAD dyn', 'Morris', 'AEE'])
    simController.runSingleCntr \
        (dwnSmple       = False,  
         modes          = ['CEDAR'], #['F2P', 'AEE', 'Morris', 'CEDAR'],
         # modes          = ['F2P', 'Morris', 'CEDAR', 'SEAD stat', 'SEAD dyn'], #'['Tetra stat', 'F2P', 'SEAD stat', 'SEAD dyn', 'CEDAR', 'Morris'] 
        # modes          = ['Morris'], #'['Tetra stat', 'F2P', 'SEAD stat', 'SEAD dyn', 'CEDAR', 'Morris'] 
        # modes          = ['CEDAR'], #'['Tetra stat', 'F2P', 'SEAD stat', 'SEAD dyn', 'CEDAR', 'Morris'] 
        # modes          = ['F2P'], #'['Tetra stat', 'F2P', 'SEAD stat', 'SEAD dyn', 'CEDAR', 'Morris'] 
        # modes          = ['SEAD stat'], #'['Tetra stat', 'F2P', 'SEAD stat', 'SEAD dyn', 'CEDAR', 'Morris'] 
        # modes          = ['SEAD dyn'], #'['Tetra stat', 'F2P', 'SEAD stat', 'SEAD dyn', 'CEDAR', 'Morris'] 
         cntrSize       = 4, 
         numOfExps      = 50,
         erTypes        = ['WrRmse', 'RdRmse'], # The error modes to gather during the simulation. Options are: 'WrEr', 'WrRmse', 'RdEr', 'RdRmse' 
         cntrMaxVal     = None, 
         )

class SimController (object):
    """
    Controller that runs a simulation 
    """

    def __init__ (self, 
                  verbose=[]): # defines which outputs would be written to .res / .pcl output files. See the VERBOSE macros as settings.py. 
        
        self.verbose = verbose
        if settings.VERBOSE_DETAILED_RES in self.verbose or settings.VERBOSE_FULL_RES in self.verbose:
            self.verbose.append (settings.VERBOSE_RES)
        if not (settings.VERBOSE_PCL in self.verbose):
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
                    if (settings.VERBOSE_DETAILS in self.verbose): 
                        print ('realVal={:.0f} oldVal={:.0f}, cntrWoScaling={:.0f}, cntrNewValScaled={:.0f}, maxRealVal={:.0f}'
                               .format (realValCntr, cntrVal, cntrAfterInc['val'], cntrNewVal, self.maxRealVal))
                    if (cntrNewVal != cntrVal): # the counter was incremented
                        cntrVal = cntrNewVal
                        # curRelativeErr = abs(realValCntr - cntrVal)/realValCntr
                        self.cntrRecord['wrEr'][expNum] += abs(realValCntr - cntrVal)/realValCntr
                        self.numOfPoints       [expNum] += 1  
                    if self.dwnSmple:
                        if cntrAfterInc['val']==self.cntrRecord['cntr'].cntrMaxVal: # the cntr overflowed --> downsample
                            self.cntrRecord['cntr'].incCntr (mult=True, factor=1/2)
                            self.cntrRecord['sampleProb'] /= 2
                        if (settings.VERBOSE_DETAILS in self.verbose): 
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
                'settingsStr'       : self.cntrRecord['cntr'].genSettingsStr(),
                'Avg'               : wrErAvg,
                'Lo'                : wrErConfInterval[0],
                'Hi'                : wrErConfInterval[1]}
        self.dumpDictToPcl      (dict, pclOutputFile)
        self.writeDictToResFile (dict)
    
    def dumpDictToPcl (self, dict, pclOutputFile):
        """
        Dump a single dict of data into pclOutputFile
        """
        if (settings.VERBOSE_PCL in self.verbose):
            pickle.dump(dict, pclOutputFile) 
    
    def writeDictToResFile (self, dict):
        """
        Write a single dict of data into resOutputFile
        """
        if (settings.VERBOSE_RES in self.verbose):
            printf (self.resFile, f'{dict}\n\n') 
    
    def runSingleCntrSingleModeWrRmse (self, pclOutputFile=None):
        """
        Run a single counter of mode self.mode (self.mode is the approximation cntr architecture - e.g., 'F2P', 'CEDAR').  
        Collect and write statistics about the write ("hit time") errors.
        "Hit time" error (aka "wr error") is the diff between the value the cntr represent, and
        the # of increments ("hit time") needed to make the cntr reach that value.
        The type of statistic collected is the Round Square Mean Error of such write errors.
        """
        
        self.cntrRecord['sumSqEr']  = [0] * self.numOfExps # self.cntrRecord['sumSqEr'][j] will hold the sum of the square errors collected at experiment j. 
        self.numOfPoints            = [0] * self.numOfExps # self.numOfPoints[j] will hold the number of points collected for statistic at experiment j. The number of points varies, as it depends upon the random process of increasing the approximated cntr. 
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
                    cntrAfterInc = self.cntrRecord['cntr'].incCntr (factor=int(1), mult=False, verbose=self.verbose)
                    cntrNewVal   = cntrAfterInc['val'] / self.cntrRecord['sampleProb']
                    if (settings.VERBOSE_DETAILS in self.verbose): 
                        print ('realVal={:.0f} oldVal={:.0f}, cntrWoScaling={:.0f}, cntrNewValScaled={:.0f}, maxRealVal={:.0f}'
                               .format (realValCntr, cntrVal, cntrAfterInc['val'], cntrNewVal, self.maxRealVal))
                    if (cntrNewVal != cntrVal): # the counter was incremented
                        cntrVal = cntrNewVal
                        # curRelativeErr = ((realValCntr - cntrVal)/realValCntr)**2
                        self.cntrRecord['sumSqEr'][expNum] += (((realValCntr - cntrVal)/realValCntr)**2)
                        self.numOfPoints          [expNum] += 1
                        if settings.VERBOSE_LOG in self.verbose:
                            printf (self.log_file, 'realValCntr={}, cntrVal={}, added sumSqEr={:.4f}\n' .format (realValCntr, cntrVal, ((realValCntr - cntrVal)/realValCntr)**2))

                    if self.dwnSmple:
                        if cntrAfterInc['val']==self.cntrRecord['cntr'].cntrMaxVal: # the cntr overflowed --> downsample
                            self.cntrRecord['cntr'].incCntr (mult=True, factor=1/2)
                            self.cntrRecord['sampleProb'] /= 2
                        if (settings.VERBOSE_DETAILS in self.verbose): 
                            print ('smplProb={}' .format (self.cntrRecord['sampleProb'])) 
                    else:
                        if cntrAfterInc['val']==self.cntrRecord['cntr'].cntrMaxVal: # the cntr reached its maximum values and no down-sample is used --> finish this experiment
                            break  
            if settings.VERBOSE_LOG in self.verbose:
                printf (self.log_file, 'sumSqEr={:.2f}\n' .format (self.cntrRecord['sumSqEr'][expNum]))
 
        dict = self.calcRmseStat ()
        self.dumpDictToPcl       (dict, pclOutputFile)
        self.writeDictToResFile  (dict)


    def runSingleCntrSingleModeRdRmse (self, pclOutputFile=None): 
        """
        Run a single counter of mode self.mode (self.mode is the approximation cntr architecture - e.g., 'F2P', 'CEDAR').  
        Collect and write statistics about the errors w.r.t. the real cntr (measured) value.
        The error is calculated upon each increment of the real cntr (measured) value, 
        as the difference between the measured value, and the value represented by the cntr.
        The type of statistic collected is the Round Square Mean Error of such write errors.
        """
    
        self.cntrRecord['sumSqEr'] = [0] * self.numOfExps # self.cntrRecord['sumSqEr'][j] will hold the sum of the square errors collected at experiment j. 
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
                    cntrAfterInc = self.cntrRecord['cntr'].incCntr (factor=int(1), mult=False, verbose=self.verbose)
                    cntrNewVal   = cntrAfterInc['val'] / self.cntrRecord['sampleProb']
                    if (settings.VERBOSE_DETAILS in self.verbose): 
                        print ('realVal={:.0f} oldVal={:.0f}, cntrWoScaling={:.0f}, cntrNewValScaled={:.0f}, maxRealVal={:.0f}'
                               .format (realValCntr, cntrVal, cntrAfterInc['val'], cntrNewVal, self.maxRealVal))
                    cntrVal = cntrNewVal
                    if (self.dwnSmple and cntrAfterInc['cntrVec']==self.cntrRecord['cntr'].cntrMaxVec): # the cntr overflowed --> downsample
                        self.cntrRecord['cntr'].incCntr (mult=True, factor=1/2)
                        self.cntrRecord['sampleProb'] /= 2
                        if (settings.VERBOSE_DETAILS in self.verbose): 
                            print ('smplProb={}' .format (self.cntrRecord['sampleProb'])) 
                self.cntrRecord['sumSqEr'][expNum] += (((realValCntr - cntrVal)/realValCntr)**2)
        if (settings.VERBOSE_LOG in self.verbose):
            printf (self.log_file, 'diff vector={}\n\n' .format (self.cntrRecord['wrErVar']))
    
        dict = self.calcRmseStat    ()
        if settings.VERBOSE_PCL in self.verbose:
            self.dumpDictToPcl          (dict, pclOutputFile)
        self.writeDictToResFile     (dict)
        
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
                    if (settings.VERBOSE_DETAILS in self.verbose): 
                        print ('realVal={:.0f} oldVal={:.0f}, cntrWoScaling={:.0f}, cntrNewValScaled={:.0f}, maxRealVal={:.0f}'
                               .format (realValCntr, cntrVal, cntrAfterInc['val'], cntrNewVal, self.maxRealVal))
                    cntrVal = cntrNewVal
                    if (self.dwnSmple and cntrAfterInc['cntrVec']==self.cntrRecord['cntr'].cntrMaxVec): # the cntr overflowed --> downsample
                        self.cntrRecord['cntr'].incCntr (mult=True, factor=1/2)
                        self.cntrRecord['sampleProb'] /= 2
                        if (settings.VERBOSE_DETAILS in self.verbose): 
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
                'settingsStr'       : self.cntrRecord['cntr'].genSettingsStr(),
                'Avg'               : rdErAvg,
                'Lo'                : rdErConfInterval[0],
                'Hi'                : rdErConfInterval[1]}
        self.dumpDictToPcl       (dict, pclOutputFile)
        self.writeDictToResFile  (dict)
        
    def calcRmseStat (self) -> dict: 
        """
        Calculate and potentially print to .log and/or .res file (based on self.verbose) the RMSE statistics based on the values measured and stored in self.cntrRecord['sumSqEr'].
        Return a dict of the calculated data.  
        """
        
        self.cntrRecord['Rmse']     = [math.sqrt (self.cntrRecord['sumSqEr'][expNum]/self.numOfPoints[expNum]) for expNum in range(self.numOfExps)]
        self.cntrRecord['normRmse'] = [              self.cntrRecord['Rmse'][expNum]/self.numOfPoints[expNum]  for expNum in range(self.numOfExps)]
        if (settings.VERBOSE_LOG in self.verbose):
            printf (self.log_file, 'normRmse=')
            printarFp (self.log_file, self.cntrRecord['normRmse'])
        
        normRmseAvg          = np.average    (self.cntrRecord['normRmse'])
        normRmseConfInterval = settings.confInterval (ar=self.cntrRecord['normRmse'], avg=normRmseAvg)
        return {'erType'        : self.erType,
                'numOfExps'     : self.numOfExps,
                'mode'          : self.cntrRecord['mode'],
                'cntrSize'      : self.cntrSize, 
                'cntrMaxVal'    : self.cntrMaxVal,
                'settingsStr'   : self.cntrRecord['cntr'].genSettingsStr(),
                'Avg'           : normRmseAvg,
                'Lo'            : normRmseConfInterval[0],
                'Hi'            : normRmseConfInterval[1]}

    def measureResolutions (self, cntrSizes, modes) -> None:  # modes (type of counter) to run  
        """
        Loop over all possible representations, measure the relative resolution, and write the results to output files as defined by self.verbose.
        """
        if settings.VERBOSE_PCL in self.verbose:
            if os.path.exists('../res/pcl_files/resolution.pcl'):
                os.remove('../res/pcl_files/resolution.pcl')
            pclOutputFile = open(f'../res/pcl_files/resolution.pcl', 'ab+')
        for self.cntrSize in cntrSizes:
            self.conf = settings.getConfByCntrSize (cntrSize=self.cntrSize)
            self.cntrMaxVal   = self.conf['cntrMaxVal'] 
            self.hyperSize    = self.conf['hyperSize'] 
            self.hyperMaxSize = self.conf['hyperMaxSize'] 
            for self.mode in modes:
                self.initCntrRecord ()
                listOfVals = []
                for i in range (2**self.cntrSize-2 if self.mode=='SEAD dyn' else (1 << self.cntrSize)):
                    cntrVec = np.binary_repr(i, self.cntrSize) 
                    listOfVals.append (self.cntrRecord['cntr'].cntr2num(cntrVec))           
                if self.mode in ['F2P', 'F3P']:
                    listOfVals = sorted (listOfVals)
                points = {'X' : listOfVals[:len(listOfVals)-1], 'Y' : [(listOfVals[i+1]-listOfVals[i])/listOfVals[i+1] for i in range (len(listOfVals)-1)]}
                if settings.VERBOSE_PCL in self.verbose:
                    self.dumpDictToPcl ({'mode' : self.mode, 'cntrSize' : self.cntrSize, 'points' : points}, pclOutputFile)

    def initCntrRecord (self):
        """
        Set self.cntrRecord, which holds the counters to run
        """
        # Set self.cntrRecord, which holds the counter to run
        if (self.mode=='F2P'):
            self.cntrRecord = {'mode' : 'F2P', 'cntr' : F2P.CntrMaster(mode='F2P', cntrSize=self.cntrSize, hyperSize=self.hyperSize, verbose=self.verbose)}
        elif (self.mode=='F3P'):
            self.cntrRecord = {'mode' : 'F3P', 'cntr' : F2P.CntrMaster(mode='F3P', cntrSize=self.cntrSize, hyperMaxSize=self.hyperMaxSize)}
        elif (self.mode=='SEAD stat'):
            self.expSize      = self.conf['seadExpSize']
            self.cntrRecord = {'mode' : self.mode, 'cntr' : SEAD.CntrMaster(mode='stat', cntrSize=self.cntrSize, expSize=self.expSize)}
        elif (self.mode=='SEAD dyn'):
            self.cntrRecord = {'mode' : self.mode, 'cntr' : SEAD.CntrMaster(mode='dyn', cntrSize=self.cntrSize)}
        elif (self.mode=='CEDAR'):
            self.cntrRecord = {'mode' : self.mode, 'cntr' : CEDAR.CntrMaster(cntrSize=self.cntrSize, cntrMaxVal=self.cntrMaxVal)}
        elif (self.mode=='Morris'):
            self.cntrRecord = {'mode' : self.mode, 'cntr' : Morris.CntrMaster(cntrSize=self.cntrSize, cntrMaxVal=self.cntrMaxVal)}
        elif (self.mode=='Tetra stat'):
            self.cntrRecord = {'mode' : self.mode, 'cntr' : TetraStatic.CntrMaster(cntrSize=self.cntrSize, tetraSize=self.conf['tetraSize'])}
        elif (self.mode=='Tetra dyn'):
            self.cntrRecord = {'mode' : self.mode, 'cntr' : TetraDynamic.CntrMaster(cntrSize=self.cntrSize, tetraMaxSize=self.conf['tetraMaxSize'])}
        elif (self.mode=='AEE'):
            self.cntrRecord = {'mode' : self.mode, 'cntr' : AEE.CntrMaster(cntrSize=self.cntrSize, cntrMaxVal=self.cntrMaxVal)}
        else:
            settings.error ('mode {} that you chose is not supported' .format (self.mode))
        


    def runSingleCntrSingleMode (self):
        """
        Run a single counter for the given mode for the requested numOfExps, and write the results (statistics
        about the absolute/relative error) to a .res file.
        """        
        if self.cntrMaxVal==None:
            self.conf = settings.getConfByCntrSize (cntrSize=self.cntrSize)
            self.cntrMaxVal   = self.conf['cntrMaxVal'] 
            self.hyperSize    = self.conf['hyperSize'] 
            self.hyperMaxSize = self.conf['hyperMaxSize'] 
                    
        
        self.initCntrRecord () # Set self.cntrRecord, which holds the counter to run
        self.maxRealVal         = self.cntrMaxVal if (self.maxRealVal==None) else self.maxRealVal
        if self.cntrRecord['cntr'].cntrMaxVal < self.maxRealVal and (not(self.dwnSmple)):
            settings.error ('The counter of type {} can reach max val={} which is smaller than the requested maxRealVal {}, and no dwn smpling was used' . format (self.cntrRecord['mode'], self.cntrRecord['cntr'].cntrMaxVal, self.maxRealVal))

        # open output files
        outputFileStr = '1cntr_{}{}' .format (self.machineStr, '_w_dwnSmpl' if self.dwnSmple else '')
        if (settings.VERBOSE_RES in self.verbose):
            self.resFile = open (f'../res/{outputFileStr}.res', 'a+')
            
        print ('Started running runSingleCntr at t={}. mode={}, cntrSize={}, maxRealVal={}, cntrMaxVal={}' .format (
                datetime.now().strftime("%H:%M:%S"), self.mode, self.cntrSize, self.maxRealVal, self.cntrRecord['cntr'].cntrMaxVal))
        
        # run the simulation          
        for self.erType in self.erTypes:
            if not (self.erType in ['WrEr', 'WrRmse', 'RdEr', 'RdRmse']):
                settings.error ('Sorry, the requested error mode {self.erType} is not supported')
            pclOutputFile = None # default value
            if settings.VERBOSE_PCL in self.verbose:
                pclOutputFile = self.openPclOuputFile (pclOutputFileName=f'{outputFileStr}_{self.erType}.pcl')
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
        If settings.VERBOSE_PCL is set, close sel.fpclOutputFile
        """
        if settings.VERBOSE_PCL in self.verbose:
            pclOutputFile.close ()

    def openPclOuputFile (self, pclOutputFileName):
        """
        If settings.VERBOSE_PCL is set, return an pclOutputFile with the requested file name.
        Else, return None
        """
        if settings.VERBOSE_PCL in self.verbose:
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
                       erTypes      = [],
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

if __name__ == '__main__':
    try: 
        main ()
        # F2P         = [0.1, 0.4, 0.6, 0.6]
        # CEDAR       = [0.1, 0.5, 0.9, 0.9]
        # mean_F2P    = mean(F2P)
        # mean_CEDAR  = mean(CEDAR)
        # Rmse_F2P    = settings.RmseOfVec (F2P)
        # Rmse_CEDAR  = settings.RmseOfVec (CEDAR)
        # print ('mean_F2P={:.2f}, mean_CEDAR={:.2f}, ratio={:.2f}' .format (mean_F2P, mean_CEDAR, mean_F2P/mean_CEDAR))
        # print ('Rmse_F2P={:.2f}, Rmse_CEDAR={:.2f}, ratio={:.2f}' .format (Rmse_F2P, Rmse_CEDAR, Rmse_F2P/Rmse_CEDAR))
    except KeyboardInterrupt:
        print('Keyboard interrupt.')
