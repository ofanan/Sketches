    def runSingleCntrSingleModeWrRmse (self):
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
 
        self.erType      = 'WrRmse'
        for rel_abs_n in [True, False]:
            for statType in ['Mse', 'normRmse']:
                dict = calcPostSimStat (
                    sumSqEr      = self.cntrRecord['sumSqRelEr'] if rel_abs_n else self.cntrRecord['sumSqAbsEr'],
                    numMeausures = self.numOfPoints,   
                    statType     = statType,
                    verbose      = self.verbose,
                    logFile      = self.logFile,
                )
                self.handleResDict (dict, rel_abs_n)