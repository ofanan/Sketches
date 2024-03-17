import os, scipy, numpy as np
# from datetime import datetime
from tictoc import tic, toc
import matplotlib.pyplot as plt

from printf import printf, printar, printarFp
import settings, ResFileParser, F2P_sr, F2P_lr, F2P_li, FP  
from SingleCntrSimulator import main, getAllValsFP, getAllValsF2P
from ResFileParser import colors, colorOfLabel, markerOfMode, MARKER_SIZE_SMALL, FONT_SIZE, LEGEND_FONT_SIZE

def clamp (vec: np.array, lowerBnd: float, upperBnd: float) -> np.array:
    """
    Clamp a the input vector vec, as follows.
    For each item in vec:
    - if x<min(grid), assign x=lowrBnd
    - if x>max(grid), assign x=upperBnd
    """
    vec[vec < lowerBnd] = lowerBnd
    vec[vec > upperBnd] = upperBnd
    return vec 

def dequantize (vec : np.array, scale : float) -> np.array:
    """
    Dequantize the given vector, namely, multiply each element in it by the given scale.
    """
    return [item*scale for item in vec] 

def calcMse (orgVec     : np.array, # vector before quantization 
             changedVec : np.array, # vector after quantization+dequantization
             dist       : str ='Gaussian', # distribution by which the MSE is weighted
             stdev      : float = 0.01,       # standard variation of the distribution; the expected value is 0.
             label      : str = None,        # a string defining the mode (e.g., 'F2P_lr'
             scale      : float = None,       # the scale by which orgVec was quantized
             logFile     = None, # object for the logFile; to be used if the verbose requests for logFile
             verbose    : list = []    # level of verbose, as defined in settings.py 
             ):
    """
    Calculate the: 
    - MSE (Mean Square Error), both relative and absolute, between the original vector and the changed vector.
    - The Mse, weighted by the given distribution and stdev (standard variation). 
    """
    if dist!='Gaussian':
        settings.error (f'In FPQuantization.calcMse(). Sorry, the distribution {dist} you chose is not supported.')
    # weightedAbsMseVec      = [scipy.stats.norm(0, stdev).pdf(orgVec[i])*(orgVec[i]-changedVec[i])**2 for i in range(len(orgVec))]
    weightedAbsMseVec = np.empty(len(orgVec))
    for i in range (10): #(len(orgVec)): 
        weightedAbsMseVec[i] = scipy.stats.norm(0, stdev).pdf(orgVec[i])
        print (f'orgVec[i]={orgVec[i]}, weightedMse={weightedAbsMseVec[i]}') #$$$
    settings.error ('reag') #$$$
    weightedRelMseVec      = np.empty(len([item for item in orgVec if item!=0]))
    idxInweightedRelMseVec = 0
    for i in range(len(orgVec)):
        if orgVec[i]==0:
            continue
        weightedRelMseVec[idxInweightedRelMseVec] = scipy.stats.norm(0, stdev).pdf(orgVec[i])*((orgVec[i]-changedVec[i])/orgVec[i])**2 
        idxInweightedRelMseVec += 1

    if settings.VERBOSE_LOG in verbose:
        printf (logFile, f'// Label={label}\n')
        for i in range (10):
             printf (logFile, f'i={i}, org={orgVec[i]}, changed={changedVec[i]}, PDF={scipy.stats.norm(0, stdev).pdf(orgVec[i])}, weightedAbsMse={weightedAbsMseVec[i]}\n')
                    
    return {
        'label'             : label,
        'scale'             : scale, 
        'avgRelMse'         : sum ([((orgVec[i]-changedVec[i])/orgVec[i])**2 for i in range(len(orgVec)) if orgVec[i]!=0]) / len(orgVec),
        'absErrVec'         : [abs(orgVec[i]-changedVec[i]) for i in range(len(orgVec))],
        'weightedAbsMseVec' : weightedAbsMseVec,
        'avgWeightedAbsMse' : np.mean (weightedAbsMseVec),
        'weightedRelMseVec' : weightedRelMseVec,
        'avgWeightedRelMse' : np.mean (weightedRelMseVec)
        }

def scaleGrid (grid : np.array, lowerBnd=0, upperBnd=100) -> np.array:
    """
    Scale the given sorted grid into the given range [lowerBnd, upperBnd]
    """
    scale = (upperBnd-lowerBnd) / (grid[-1]-grid[0])
    return [item*scale for item in grid] 
    
    
def plotScaledGrids (
        modes       = [], # modes to be simulated, e.g. FP, F2P_sr. 
        cntrSize    = 6,  # of bits, including the sign bit 
        hyperSize   = 2,  # size of the hyper-exp, when simulating F2P  
        signed      = False, # when True, plot also negative values (symmetrically w.r.t. 0).
        verbose     = []  # level of verbose, as defined in settings.py. 
        ) -> None:
    """
    """
    # setPltParams = lambda self, size = 'large': matplotlib.rcParams.update({
    #     'font.size'         : FONT_SIZE,
    #     'legend.fontsize'   : LEGEND_FONT_SIZE,
    #     'xtick.labelsize'   : FONT_SIZE,
    #     'ytick.labelsize'   : FONT_SIZE,
    #     'axes.labelsize'    : FONT_SIZE,
    #     'axes.titlesize'    : FONT_SIZE, }) if (size == 'large') else matplotlib.rcParams.update({
    #     'font.size'         : FONT_SIZE_SMALL,
    #     'legend.fontsize'   : LEGEND_FONT_SIZE_SMALL,
    #     'xtick.labelsize'   : FONT_SIZE_SMALL,
    #     'ytick.labelsize'   : FONT_SIZE_SMALL,
    #     'axes.labelsize'    : FONT_SIZE_SMALL,
    #     'axes.titlesize'    : FONT_SIZE_SMALL
    #     })
    _, ax = plt.subplots()
    resRecords = []
    lenGrid     = 2**cntrSize
    for mode in modes:
        if mode.startswith('FP'):
            expSize = int(mode.split ('_e')[1])
            mode = 'FP'
            resRecords.append ({
                'mode'  : mode,
                'label' : ResFileParser.genFpLabel(expSize=expSize, mantSize=cntrSize-expSize),
                'grid'  : scaleGrid (getAllValsFP(cntrSize=cntrSize, expSize=expSize, verbose=verbose, signed=signed)), 
                })
        elif mode.startswith('F2P'):
            flavor = mode.split('_')[1]
            resRecords.append ({
                'mode'  : mode,
                'label' : ResFileParser.genF2pLabel(flavor=flavor),
                'grid'  : scaleGrid (getAllValsF2P (flavor=flavor, cntrSize=cntrSize, hyperSize=hyperSize, verbose=verbose, signed=signed)) 
                })

    for i in range(len(resRecords)): 
        resRecord = resRecords[i]     
        ax.plot (resRecord['grid'], 
                 [len(resRecords)-i for item in range(lenGrid)], 
                 color      = colorOfLabel[resRecord['label']], 
                 marker     = markerOfMode[mode], 
                 linestyle  = 'None', 
                 markersize = MARKER_SIZE_SMALL, 
                 label      = resRecord['label'])  # Plot the conf' interval line
    frame1 = plt.gca()
    frame1.axes.get_yaxis().set_visible(False)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend (by_label.values(), by_label.keys(), fontsize=LEGEND_FONT_SIZE, frameon=False)
    plt.xlim (0,5)        
    plt.show()

def quantize (vec  : np.array, # The vector to quantize 
              grid : np.array  # The quantization grid (all the values that can be represented by the destination number representation
              ) -> [np.array, float]: # [the_quantized_vector, the scale_factor (by which the vector was divided)] 
    """
    Quantize an input vector, using symmetric Min-max quantization. 
    This is done by:
    - Quantizing the vector, namely:
      - Clamping and scaling the vector. The scaling method is minMax.
      - Rounding the vector to the nearest values in the grid.
    """
    upperBnd    = max (abs(vec[0]), abs(vec[-1])) # The upper bound is the largest absolute value in the vector to quantize.
    lowerBnd    = -upperBnd
    scaledVec   = clamp (vec, lowerBnd, upperBnd)
    scale       = (upperBnd-lowerBnd) / (grid[-1]-grid[0])
    scaledVec   = [item/scale for item in vec] # The vector after scaling and clamping (still w/o rounding)  
    quantVec    = np.empty (len(vec)) # The quantized vector (after rounding scaledVec) 
    idxInGrid = int(0)
    for idxInVec in range(len(scaledVec)):
        if idxInGrid==len(grid): # already reached the max grid val --> all next items in q should be the last item in the grid 
            quantVec[idxInVec] = grid[-1]
            continue
        quantVec[idxInVec]= grid[idxInGrid]
        minAbsErr = abs (scaledVec[idxInVec]-quantVec[idxInVec])
        while (idxInGrid < len(grid)):
            quantVec[idxInVec]= grid[idxInGrid]
            absErr = abs (scaledVec[idxInVec]-quantVec[idxInVec])
            if absErr <= minAbsErr:
                minAbsErr = absErr
                idxInGrid += 1
            else:
               idxInGrid -= 1
               quantVec[idxInVec]= grid[idxInGrid]
               break
    return [quantVec, scale]

def genVec2Quantize (dist       : str   = 'uniform',  # distribution from which points are drawn  
                     lowerBnd   : float = 0,   # lower bound for the generated points  
                     upperBnd   : float = 10,   # upper bound for the generated points
                     stdev      : float = 1,   # standard variation when generating a Gaussian dist' points
                     numPts     : int   = 1000, # Num of points in the generated vector
                     ) -> np.array:
    """
    Generate a vector to be quantized, using the requested distribution.
    """
    if dist=='Uniform':
        return np.array([(lowerBnd + i*(upperBnd-lowerBnd)/numPts) for i in range(numPts)])
    elif dist=='Gaussian':
        return (np.sort (np.random.randn(numPts) * stdev))
    else:
        settings.error ('In Quantization.genVec2Quantize(). Sorry. The distribution {dist} you chose is not supported.')
    
    
def simQuantErr (modes          = [], # modes to be simulated, e.g. FP, F2P_sr. 
                 cntrSize       = 8,  # of bits, including the sign bit 
                 expSizes       = [1], # size of the exponent when simulating FP 
                 hyperSize      = 2,  # size of the hyper-exp, when simulating F2P  
                 numPts         = 1000, # num of points in the quantized vec
                 verbose        = [],  # level of verbose, as defined in settings.py.
                 stdev          = 1,   # standard variation of the vector to quantize, when drawn from a Gaussian dist'  
                 vecLowerBnd    = -float('inf'), # lower Bnd of the generated vector to quantize, if drawn from a uniform dist'  
                 vecUpperBnd    = float('inf')   # upper Bnd of the generated vector to quantize, if drawn from a uniform dist'
                 ):
    """
    Simulate the required configuration and output the results (the quantization errors) as defined by the verbose.
    """
    np.random.seed (settings.SEED)
    if settings.VERBOSE_RES in verbose:
        resFile = open (f'../res/quant_n{cntrSize}.res', 'a+')
    if settings.VERBOSE_LOG in verbose:
        logFile = open (f'../res/quant_n{cntrSize}.log', 'w')        
    vec2quantize = genVec2Quantize (
        dist        = 'Uniform', 
        lowerBnd    = vecLowerBnd,   # lower bound for the generated points  
        upperBnd    = vecUpperBnd,   # upper bound for the generated points
        stdev       = stdev, 
        numPts      = numPts)
    _, ax = plt.subplots()
    resRecords = []
    for mode in modes:
        if mode=='FP':
            for expSize in expSizes: 
                grid     = getAllValsFP(cntrSize=cntrSize, expSize=expSize, verbose=[], signed=True)
                [quantizedVec, scale] = quantize(vec=vec2quantize, grid=grid)
                dequantizedVec = dequantize(vec=quantizedVec, scale=scale)
                label = ResFileParser.genFpLabel(expSize=expSize, mantSize=cntrSize-1-expSize),
                # print (f'vec2quant={vec2quantize}\ndeqVec={dequantizedVec}') #$$$
                resRecords.append (calcMse(
                        orgVec      = vec2quantize, 
                        changedVec  = dequantizedVec, 
                        label       = label,
                        stdev       = stdev,
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
            resRecords.append (calcMse(
                    orgVec      = vec2quantize, 
                    changedVec  = dequantizedVec, 
                    label       = ResFileParser.genF2pLabel(flavor=flavor),
                    scale       = scale,
                    stdev       = stdev,
                    logFile     = logFile,
                    verbose     = verbose
                    ))
        elif mode=='shortTest':
            grid = np.array([i for i in range(-10, 11)])
            vec2quantize = np.array([-100, -95, -7, 99, 100])
            [quantizedVec, scale] = quantize(vec=vec2quantize, grid=grid)
            dequantizedVec = dequantize(vec=quantizedVec, scale=scale)
            resRecords.append (calcMse(
                    orgVec      = vec2quantize, 
                    changedVec  = dequantizedVec, 
                    label       = 'shortTest'
                    ))
        else:
            settings.error ('Sorry, the requested mode {mode} is not supported.')

    if settings.VERBOSE_COUT_CNTRLINE in verbose:
        print (resRecords)
        
    if settings.VERBOSE_RES in verbose:
        for resRecord in resRecords:
            for key, value in resRecord.items():
                if not key.endswith('Vec'):
                    printf (resFile, f'{key} : {value}\n')
            printf (resFile, '\n\n')

    if settings.VERBOSE_PLOT not in verbose:
        return
     
    for i in range(len(resRecords)): 
        resRecord = resRecords[i]
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


    

# plotScaledGrids (cntrSize=6, modes=['FP_e1', 'F2P_sr', 'FP_e5', 'F2P_lr'])
stdev = 1
simQuantErr (modes          = ['F2P_sr','FP'], #   
             expSizes       = [1], 
             numPts         = 1000, 
             stdev          = stdev,
             vecLowerBnd    = -1*stdev,
             vecUpperBnd    =  1*stdev,
             verbose= [settings.VERBOSE_LOG]) #[settings.VERBOSE_RES, settings.VERBOSE_PLOT])  
# print (scipy.stats.norm(0, stdev).pdf(-0.982))