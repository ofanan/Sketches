    def runSingleCntrSingleModeWrEr (self):
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
 
        if (VERBOSE_LOG in self.verbose):
            printf (self.log_file, f'diff vector={self.cntrRecorwrErimeVar}\n\n')

        self.cntrRecord['wrEr'] = np.divide (self.cntrRecord['wrEr'], self.numOfPoints) 
        if (VERBOSE_LOG in self.verbose):
            printf (self.log_file, 'wrEr=\n{:.3f}\n, ' .format (self.cntrRecord['wrEr']))
        
        wrErAvg             = np.average    (self.cntrRecord['wrEr'])
        wrErConfInterval = confInterval (ar=self.cntrRecord['wrEr'], avg=wrErAvg, confLvl=self.confLvl)
        dict = {'erType'            : self.erType,
                'numOfExps'         : self.numOfExps,
                'mode'              : self.cntrRecord['mode'],
                'cntrSize'          : self.cntrSize, 
                'cntrMaxVal'        : self.cntrMaxVal,
                'settingStr'        : self.cntrRecord['cntr'].genSettingsStr(),
                'Avg'               : wrErAvg,
                'Lo'                : wrErConfInterval[0],
                'Hi'                : wrErConfInterval[1],
                'confLvl'           : self.confLvl
                }
        self.dumpDictToPcl      (dict)
        self.writeDictToResFile (dict)
