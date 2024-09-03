"""
Controller that runs single-counter simulations, using various types of counters. 
"""
import os, math, pickle, time, random #sys
from printf import printf, printar, printarFp
import numpy as np #, scipy.stats as st, pandas as pd
import settings, Cntr, CEDAR, Morris, AEE, FP, SEAD_stat, SEAD_dyn   
import F2P_sr, F2P_lr, F2P_li, F2P_li_ds, F2P_si, F3P_sr, F3P_lr, F3P_li, F3P_li_ds, F3P_si    
from settings import * 
from datetime import datetime
np.set_printoptions(precision=2)

class SingleCntrSimulator (object):
    """
    Controller that runs single-counter simulations, using various types of counters and configurations. 
    """

    def __init__ (
            self, 
            seed    = SEED,
            verbose = [] # defines which outputs would be written to .res / .pcl output files. See the VERBOSE macros as py.
        ):  
        
        self.seed    = seed
        random.seed (self.seed)
        
        self.confLvl = 0.95 # Required confidence level at the conf' interval calculation.
        self.verbose = verbose
        if VERBOSE_DETAILED_RES in self.verbose:
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
        if not (VERBOSE_PROGRESS in self.verbose):
            return
        if infoStr==None:
            printf (self.log_file, f'starting experiment{expNum}\n')
        else:
            printf (self.log_file, f'{infoStr}\n')
    
   
    def dumpDictToPcl (self, dict):
        """
        Dump a single dict of data into pclOutputFile
        """
        if (VERBOSE_PCL in self.verbose):
            pickle.dump(dict, self.pclOutputFile) 
    
    def writeDictToResFile (self, dict):
        """
        Write a single dict of data into resOutputFile
        """
        if (VERBOSE_RES in self.verbose):
            printf (self.resFile, f'{dict}\n\n') 
    
    def runSingleCntrSingleModeWrEr (self):
        """
        Run a single counter of mode self.mode (self.mode is the approximation cntr architecture - e.g., 'F2P', 'CEDAR').  
        Collect and write statistics about the write ("hit time") errors.
        "Hit time" error (aka "wr error") is the diff between the value the cntr represent, and
        the # of increments ("hit time") needed to make the cntr reach that value.
        The type of statistic collected is the Round Square Mean Error of such write errors.
        """
        self.cntrRecord['sumSqAbsEr'] = np.zeros (self.numOfExps) # self.cntrRecord['sumSqAbsEr'][j] will hold the sum of the square absolute errors collected at experiment j. 
        self.cntrRecord['sumSqRelEr'] = np.zeros (self.numOfExps) # self.cntrRecord['sumSqRelEr'][j] will hold the sum of the square relative errors collected at experiment j. 
        self.numOfPoints              = np.zeros (self.numOfExps) # self.numOfPoints[j] will hold the number of points collected for statistic at experiment j. The number of points varies, as it depends upon the random process of increasing the approximated cntr. 
        for expNum in range(self.numOfExps):
            if VERBOSE_LOG in self.verbose:
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
                        if VERBOSE_LOG in self.verbose:
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
            if rel_abs_n: 
                sumSqEr = self.cntrRecord['sumSqRelEr']
            else:
                sumSqEr = self.cntrRecord['sumSqAbsEr'],
            for statType in ['Mse', 'normRmse']:
                dict = calcPostSimStat (
                    sumSqEr      = sumSqEr,
                    numMeausures = self.numOfPoints,   
                    statType     = statType,
                    verbose      = self.verbose,
                    logFile      = self.logFile,
                )
                self.handleResDict (dict, rel_abs_n)


    def runSingleCntrSingleModeRdEr (self): 
        """
        Run a single counter of mode self.mode (self.mode is the approximation cntr architecture - e.g., 'F2P', 'CEDAR').  
        Collect and write statistics about the errors w.r.t. the real cntr (measured) value.
        The error is calculated upon each increment of the real cntr (measured) value, 
        as the difference between the measured value, and the value represented by the cntr.
        The type of statistic collected is the Round Mean Square Error of such write errors.
        """
        self.cntrRecord['sumSqAbsEr'] = np.zeros (self.numOfExps) # self.cntrRecord['sumSqAbsEr'][j] will hold the sum of the square absolute errors collected at experiment j. 
        self.cntrRecord['sumSqRelEr'] = np.zeros (self.numOfExps) # self.cntrRecord['sumSqRelEr'][j] will hold the sum of the square relative errors collected at experiment j. 
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
            if rel_abs_n: 
                sumSqEr = self.cntrRecord['sumSqRelEr']
            else:
                sumSqEr = self.cntrRecord['sumSqAbsEr'],
            for statType in ['Mse', 'normRmse']:
                dict = calcPostSimStat (
                    numMeausures = self.maxRealVal * np.ones(self.numOfExps), # numMeausures[j] captures the # of points collected for statistic at experiment j. In other sim settings, this # may vary, as it depends upon the random process of increasing the approximated cntr.   
                    sumSqEr      = sumSqEr,
                    statType     = statType,
                    verbose      = self.verbose,
                    logFile      = self.logFile,
                )
                self.handleResDict (dict, rel_abs_n)
        
    def handleResDict (
            self, 
            dict,       # dictionary to write to .pcl / .res file 
            rel_abs_n   # indicates whether the results are for relative or abs error
        ):

        dict['rel_abs_n']   = rel_abs_n
        dict['erType']      = self.erType
        dict['numOfExps']   = self.numOfExps
        dict['mode']        = self.cntrRecord['mode']
        dict['settingStr']  = self.cntrRecord['cntr'].genSettingsStr()
        dict['cntrSize']    = self.cntrSize
        dict['cntrMaxVal']  = self.cntrMaxVal                
        self.dumpDictToPcl       (dict)
        self.writeDictToResFile  (dict)
    
    def measureResolutionsByModes (
            self, 
            delPrevPcl  : bool = False, # When True, delete the previous .pcl file, if exists
            cntrSizes   : list = [], 
            expSize     : list = None, # num of bits in the exponent to run
            maxValBy    : str  = None,  
            modes       : list = [], # modes (type of counter) to run
            ) -> None:    
        """
        Loop over all requested modes and cntrSizes, measure the relative resolution, and write the results to output files as defined by self.verbose.
        """
        if VERBOSE_PCL in self.verbose:
            pclOutputFileName = 'resolutionByModes'
            if delPrevPcl and os.path.exists(f'../res/pcl_files/{pclOutputFileName}.pcl'):
                os.remove(f'../res/pcl_files/{pclOutputFileName}.pcl')
            self.pclOutputFile = open(f'../res/pcl_files/{pclOutputFileName}.pcl', 'ab+')
        if VERBOSE_RES in self.verbose:
            resFileName  = 'resolutionByModes'
            self.resFile = open(f'../res/{resFileName}.res', 'ab+')
        for self.cntrSize in cntrSizes:
            self.cntrMaxVal   = getCntrMaxValFromFxpStr(cntrSize=self.cntrSize, fxpSettingStr=maxValBy)                
            for self.mode in modes:
                self.genCntrRecord (expSize=None if self.mode.startswith('SEAD_stat') else expSize)
                listOfVals = np.empty (2**self.cntrSize)
                for i in range (2**self.cntrSize-2 if self.mode.startswith('SEAD_dyn') else (1 << self.cntrSize)):
                    cntrVec = np.binary_repr(i, self.cntrSize) 
                    listOfVals[i] = (self.cntrRecord['cntr'].cntr2num(cntrVec))           
                listOfVals = np.sort (listOfVals)
                print (f'mode={self.mode}, maxVal={listOfVals[-1]}')
                zeroEntries = np.where (listOfVals[1:]==0)[0]
                if len(zeroEntries)>0:
                    error (f'mode={self.mode}: a zero entry in the divisor in entries\n{zeroEntries}.Divisor is\n{listOfVals[1:]}')
                points = {'X' : listOfVals[:len(listOfVals)-1], 'Y' : np.divide (listOfVals[1:] - listOfVals[:-1], listOfVals[1:])}
                dict = {'mode' : self.mode, 'cntrSize' : self.cntrSize, 'points' : points}
                if VERBOSE_PCL in self.verbose:
                    self.dumpDictToPcl (dict)                   

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
            params = extractParamsFromSettingStr (settingStr)
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
                self.dumpDictToPcl ({'settingStr' : settingStr, 'points' : points})

    def genCntrRecord (
            self,
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
                error ('In SingleCntrSimulator.genCntrRecord(). For generating an FP.CntrMaster you must specify an expSize')
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
        elif (self.mode=='AEE'):
            self.cntrRecord = {'mode' : self.mode, 'cntr' : AEE.CntrMaster(cntrSize=self.cntrSize, cntrMaxVal=self.cntrMaxVal)}
        else:
            error ('mode {} that you chose is not supported' .format (self.mode))


    def runSingleCntrSingleMode (
        self, 
        cntrSize, 
        mode         = [], # modes (type of counter) to run. E.g., 'F2P_li_h2', 'Morris'  
        maxRealVal   = None, # The maximal value to be counted at each experiment, possibly using down-sampling. When None (default), will be equal to cntrMaxval.
        cntrMaxVal   = None, # cntrMaxVal - The maximal value that the cntr can represent w/o down-sampling. When None (default), take cntrMaxVal from Confs.  global parameter (found in this file). 
        expSize      = None, # Size of the exponent. Relevant only for Static SEAD counter. If cntrMaxVal==None (default), take expSize from Confs global parameter (found in this file). 
        numOfExps    = 1,    # number of experiments to run. 
        dwnSmple     = False,# When True, down-sample each time the counter's maximum value is reached.
        erTypes      = [],   # either 'RdEr' or 'WrEr'
        rel_abs_n    = True  # When True, consider rel err. Else, consider abs err.
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
        if (VERBOSE_DETAILED_LOG in self.verbose): # a detailed log include also all the prints of a simple log
            verbose.append(VERBOSE_LOG)
        if cntrMaxVal==None:
            self.conf         = getConfByCntrSize (cntrSize=self.cntrSize)
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
            if not (self.erType in ['WrEr', 'RdEr']):
                error (f'Sorry, the requested error type {self.erType} is not supported')
            self.pclOutputFile = None # default value
            if VERBOSE_PCL in self.verbose:
                self.pclOutputFile = open(f'../res/pcl_files/{outputFileStr}.pcl', 'ab+')
            simT = time.time()
            infoStr = '{}_{}' .format (self.cntrRecord['cntr'].genSettingsStr(), self.erType)
            self.logFile  = None
            if (VERBOSE_LOG in self.verbose or VERBOSE_PROGRESS in self.verbose):
                self.log_file = open (f'../res/log_files/{infoStr}.log', 'w')
            self.writeProgress (infoStr=infoStr)
            getattr (self, f'runSingleCntrSingleMode{self.erType}') () # Call the corresponding function, according to erType (read/write error, regular/RMSE).
            self.closePclOuputFile()
            print ('finished. Elapsed time={:.2f} secs' .format (time.time() - simT))

    def closePclOuputFile (self):
        """
        If VERBOSE_PCL is set, close sel.fpclOutputFile
        """
        if VERBOSE_PCL in self.verbose:
            self.pclOutputFile.close ()

def genCntrMasterFxp (
        cntrSize        : int, 
        numCntrs        : int = 1,
        fxpSettingStr   : str  = None,
        hyperSize       : int  = None, 
        nSystem         : str  = None, # 'F2P' or 'F3P
        flavor          : str  = None, # 'sr' / 'lr' / 'si' / 'li' 
        dwnSmpl         : bool = False, 
        verbose         : list = []    # list of verboses taken from the veroses, defined in py
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
            error (f'In SingleCntrSimulator.genCntrMasterFxp(). the requested F2P flavor {flavor} is not supported.')

    elif nSystem=='F3P':

        if flavor=='sr':
            return F3P_sr.CntrMaster(cntrSize=cntrSize, numCntrs=numCntrs, hyperMaxSize=hyperSize, verbose=verbose)
        elif flavor=='lr':
            return F3P_lr.CntrMaster(cntrSize=cntrSize, numCntrs=numCntrs, hyperMaxSize=hyperSize, verbose=verbose)
        elif flavor=='li':
            if dwnSmpl:
                return F3P_li_ds.CntrMaster(cntrSize=cntrSize, numCntrs=numCntrs, hyperMaxSize=hyperSize, verbose=verbose)
            else:
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
                  verbose   = [] # verbose level. See py
                  ):
    """
    Loop over all the binary combinations of the given counter size.
    For each combination, get the respective counter.
    Sort by an increasing value.
    Output is according to the verbose, as defined in py. In particular: 
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
        outputFile = open('../res/single_cntr_log_files/{}.res'.format(myCntrMaster.genSettingsStr()), 'w')
        printf (outputFile, f'// bias={myCntrMaster.bias}\n')
        for item in listOfVals:
            printf(outputFile, '{}={}\n'.format(item['cntrVec'], item['val']))

    listOfVals = [item['val'] for item in listOfVals]    
    if signed:
        listOfVals = makeSymmetricVec (listOfVals)
        
    return listOfVals


def getAllValsFxp (
   fxpSettingStr : str,
   verbose       : list = [], #verbose level. See py for details.
   cntrSize      : int  = 8,
   signed        : bool = False # When True, assume an additional bit for the  
):
    """
    Loop over all the binary combinations of the given counter size. 
    For each combination, get the respective counter.
    Sort by an increasing value.
    Output is according to the verbose, as defined in py. In particular: 
    If the verbose include VERBOSE_RES, print to an output file the list of cntrVecs and respective values. 
    Return the (sorted) list of values.
    """
    if signed: 
        cntrSize -= 1 
    cntrSettings = getFxpSettings (fxpSettingStr)
    hyperSize   = cntrSettings['hyperSize']
    flavor      = cntrSettings['flavor']
    myCntrMaster = genCntrMasterFxp (
        cntrSize        = cntrSize, 
        fxpSettingStr   = fxpSettingStr, 
    )
    if myCntrMaster.isFeasible==False:
        error (f'The requested configuration is not feasible.')
    listOfVals = []
    for i in range (2**cntrSize):
        cntr = np.binary_repr(i, cntrSize) 
        val = myCntrMaster.cntr2num(cntr=cntr)
        if flavor in ['si', 'li']:
            val = int(val)
        listOfVals.append ({'cntrVec' : cntr, 'val' : val})
    listOfVals = sorted (listOfVals, key=lambda item : item['val'])
    
    if (VERBOSE_RES in verbose):
        outputFile    = open ('../res/single_cntr_log_files/{}.res' .format (myCntrMaster.genSettingsStr()), 'w')
        for item in listOfVals:
            printf (outputFile, '{}={}\n' .format (item['cntrVec'], item['val']))
    
    if (VERBOSE_PCL in verbose):
        with open('../res/pcl_files/{}.pcl' .format (myCntrMaster.genSettingsStr()), 'wb') as self.pclOutputFile:
            pickle.dump(listOfVals)

    listOfVals = [item['val'] for item in listOfVals]    
    if signed:
        listOfVals = makeSymmetricVec (listOfVals)
        
    return listOfVals

def getCntrsMaxValsFxp (
        nSystem         : str, # either 'F2p' or 'F3P.
        flavor          : str  = 'sr', 
        hyperSizeRange  : list = [], # list of hyper-sizes to consider  
        cntrSizeRange   : list = [], # list of cntrSizes to consider
        verbose         : list =[VERBOSE_RES]
        ):
    """
    Get the maximum value a cntr reach for several "configurations" -- namely, all combinations of cntrSize and hyperSize.
    Print the result if was requested by the VERBOSE parameter.
    Returns the cntrMaxVal of the last conf' it was called with  
    """

    for cntrSize in cntrSizeRange:
        for hyperSize in range (1,cntrSize-2) if hyperSizeRange==None else hyperSizeRange:
            myCntrMaster = genCntrMasterFxp (
                fxpSettingStr = f'{nSystem}_{flavor}_h{hyperSize}', #$$$
                cntrSize    = cntrSize, 
            )
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

def getCntrMaxValFromFxpStr (
        cntrSize : int,
        fxpSettingStr : str
        ) -> float:
    """
    Given a string detailing the settings an F2P/F3P counter, returns its maximum representable value. 
    """
    if not(fxpSettingStr.startswith('F2P')) and not(fxpSettingStr.startswith('F3P')):
        error (f'SingleCntrSimulator.getCntrMaxValFromFxpStr() was called with Fxp settings str={fxpSettingStr}')
    myCntrMaster = genCntrMasterFxp (
        cntrSize        = cntrSize, 
        fxpSettingStr   = fxpSettingStr)
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

def testDwnSmpling ():
    """
    Test the down-sampling.
    """
    cntrSize        = 8
    fxpSettingStr   = 'F2P_li_h2_ds'
    cntrMaster = genCntrMasterFxp(
        cntrSize        = cntrSize, 
        fxpSettingStr   = fxpSettingStr, 
        verbose         = [VERBOSE_LOG_DWN_SMPL]
    ) 
    logFile = open (f'../res/log_files/{fxpSettingStr}_n{cntrSize}.log', 'w')
    cntrMaster.setLogFile (logFile)
    for i in range (2**cntrSize):
        cntrMaster.cntrs[0] = np.binary_repr(i, cntrSize)
        cntrMaster.upScale  ()


def main ():
    # maxValBy = 'F3P_li_h2'
    # modes = ['F3P_li_h2_ds'] #['AEE', 'CEDAR', 'Morris', 'SEAD_dyn']
    # for cntrSize in [6]:
    #     simController = SingleCntrSimulator (
    #         verbose = [VERBOSE_RES]
    #     ) 
    #     numSettings = getCntrMaxValFromFxpStr (cntrSize=cntrSize, fxpSettingStr=maxValBy) 
    #     for mode in modes:
    #         simController.runSingleCntrSingleMode \
    #             (dwnSmple       = False,  
    #             mode            = mode, 
    #             cntrSize        = cntrSize, 
    #             cntrMaxVal      = cntrMaxVal,
    #             numOfExps       = 1, #100,
    #             erTypes         = ['WrEr'], # Options are: 'WrEr', 'WrRmse', 'RdEr', 'RdRmse' 
    #         )
        
        simController = SingleCntrSimulator (verbose = [VERBOSE_PCL]) #VERBOSE_RES, VERBOSE_PCL],)
        maxValBy      = 'F2P_li_h2'
        simController.measureResolutionsByModes (
            cntrSizes   = [8], 
            expSize     = 2, 
            maxValBy    = maxValBy,
            modes       = [maxValBy, 'Morris', 'AEE', 'CEDAR'],            
            delPrevPcl  = True
        ) 
        # simController.measureResolutionsBySettingStrs (
        #     delPrevPcl  = False, # When True, delete the previous .pcl file, if exists
        #     settingStrs = ['FP_n15_m10_e5'], # 'FP_n15_m2_e13'],  # 'FP_n7_m5_e2'# Concrete settings for which the measurements will be done 
        #     # settingStrs = ['F2Plr_n15_h2', 'F2Psr_n15_h2'],  # 'FP_n7_m5_e2'# Concrete settings for which the measurements will be done 
        #     # settingStrs = ['F2Plr_n7_h1', 'F2Psr_n7_h1'],  # 'FP_n7_m5_e2'# Concrete settings for which the measurements will be done 
        #     )    
    

if __name__ == '__main__':
    try:
        getAllValsFxp (
            fxpSettingStr   = 'F3P_si_h2',
            cntrSize        = 5, # size of the counter, WITHOUT the sign bit (if exists).  
            signed          = False,
            verbose         = [VERBOSE_RES] #verbose level. See py for details.
        )
        # main ()
    except KeyboardInterrupt:
        print('Keyboard interrupt.')
        exit ()
        
