class SimQuantErr (ojbect):
    """
    Class for quantization simulations
    """
    
    def __init__ (
        self,
        modes          = [], # modes to be simulated, e.g. FP, F2P_sr. 
        cntrSize       = 8,  # of bits, including the sign bit 
        expSizes       = [1], # size of the exponent when simulating FP 
        hyperSize      = 2,  # size of the hyper-exp, when simulating F2P  
        numPts         = 1000, # num of points in the quantized vec
        verbose        = [],  # level of verbose, as defined in settings.py.
        stdev          = 1,   # standard variation of the vector to quantize, when drawn from a Gaussian dist'  
        vecLowerBnd    = -float('inf'), # lower Bnd of the generated vector to quantize, if drawn from a uniform dist'  
        vecUpperBnd    = float('inf')):   # upper Bnd of the generated vector to quantize, if drawn from a uniform dist'

        np.random.seed (settings.SEED)
        self.verbose = verbose
        if settings.VERBOSE_RES in self.verbose:
            self.resFile = open (f'../res/quant_n{cntrSize}.res', 'a+')
        if settings.VERBOSE_LOG in self.verobse:
            self.logFile = open (f'../res/quant_n{cntrSize}.log', 'a+')        
        vec2quantize = genVec2Quantize (
            dist        = 'Uniform', 
            lowerBnd    = vecLowerBnd,   # lower bound for the generated points  
            upperBnd    = vecUpperBnd,   # upper bound for the generated points
            stdev       = stdev, 
            numPts      = numPts)
        _, ax = plt.subplots()
        self.resRecords = []
        for mode in modes:
            if mode=='FP':
                for expSize in expSizes: 
                    grid     = getAllValsFP(cntrSize=cntrSize, expSize=expSize, verbose=[], signed=True)
                    [quantizedVec, scale] = quantize(vec=vec2quantize, grid=grid)
                    dequantizedVec = dequantize(vec=quantizedVec, scale=scale)
                    # print (f'vec2quant={vec2quantize}\ndeqVec={dequantizedVec}') #$$$
                    self.resRecords.append (calcMse(
                            orgVec      = vec2quantize, 
                            changedVec  = dequantizedVec, 
                            label       = ResFileParser.genFpLabel(expSize=expSize, mantSize=cntrSize-1-expSize),
                            scale       = scale,
                            logFile     = logFile,
                            verbose     = verbose
                            ))
            elif mode.startswith('F2P'):
                flavor = mode.split('_')[1]
                grid = getAllValsF2P (flavor=flavor, cntrSize=cntrSize, hyperSize=hyperSize, verbose=[], signed=True)
                [quantizedVec, scale] = quantize(vec=vec2quantize, grid=grid)                
                dequantizedVec = dequantize(vec=quantizedVec, scale=scale)
                # print (f'vec2quant={vec2quantize}\ndeqVec={dequantizedVec}') #$$$
                self.resRecords.append (calcMse(
                        orgVec      = vec2quantize, 
                        changedVec  = dequantizedVec, 
                        label       = ResFileParser.genF2pLabel(flavor=flavor),
                        scale       = scale
                        ))
            elif mode=='shortTest':
                grid = np.array([i for i in range(-10, 11)])
                vec2quantize = np.array([-100, -95, -7, 99, 100])
                [quantizedVec, scale] = quantize(vec=vec2quantize, grid=grid)
                dequantizedVec = dequantize(vec=quantizedVec, scale=scale)
                self.resRecords.append (calcMse(
                        orgVec      = vec2quantize, 
                        changedVec  = dequantizedVec, 
                        label       = 'shortTest'
                        ))
            else:
                settings.error ('Sorry, the requested mode {mode} is not supported.')
    
        if settings.VERBOSE_COUT_CNTRLINE in verbose:
            print (self.resRecords)
            
        if settings.VERBOSE_RES in verbose:
            for resRecord in self.resRecords:
                for key, value in resRecord.items():
                    if not key.endswith('Vec'):
                        printf (resFile, f'{key} : {value}\n')
                printf (resFile, '\n\n')
    
        if settings.VERBOSE_PLOT not in verbose:
            return
         
        for i in range(len(self.resRecords)): 
            resRecord = self.resRecords[i]
            ax.plot (vec2quantize, 
                     resRecord['weightedAbsMseVec'], 
                     color      = colorOfLabel[resRecord['label']], 
                     marker     = markerOfMode[mode], 
                    linestyle  = 'None', 
                     markersize = 2, 
                     label      = resRecord['label'])  # Plot the conf' interval line
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend (by_label.values(), by_label.keys(), fontsize=LEGEND_FONT_SIZE, frameon=False)
        # plt.yscale ('log')
        # plt.ylim (10**(-30), 10**(-2))
        plt.xlim (-1, 1)
        plt.show()