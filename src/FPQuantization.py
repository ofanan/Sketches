import numpy as np
import os
# from datetime import datetime
from tictoc import tic, toc
import matplotlib.pyplot as plt

from printf import printf, printar, printarFp
import settings, ResFileParser, F2P_sr, F2P_lr, F2P_li, FP  
from SingleCntrSimulator import main, getAllValsFP, getAllValsF2P
from ResFileParser import colors, colorOfMode, markerOfMode, MARKER_SIZE_SMALL, FONT_SIZE, LEGEND_FONT_SIZE

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
    setPltParams = lambda self, size = 'large': matplotlib.rcParams.update({
        'font.size'         : FONT_SIZE,
        'legend.fontsize'   : LEGEND_FONT_SIZE,
        'xtick.labelsize'   : FONT_SIZE,
        'ytick.labelsize'   : FONT_SIZE,
        'axes.labelsize'    : FONT_SIZE,
        'axes.titlesize'    : FONT_SIZE, }) if (size == 'large') else matplotlib.rcParams.update({
        'font.size'         : FONT_SIZE_SMALL,
        'legend.fontsize'   : LEGEND_FONT_SIZE_SMALL,
        'xtick.labelsize'   : FONT_SIZE_SMALL,
        'ytick.labelsize'   : FONT_SIZE_SMALL,
        'axes.labelsize'    : FONT_SIZE_SMALL,
        'axes.titlesize'    :FONT_SIZE_SMALL
        })
    _, ax = plt.subplots()
    plotRecords = []
    lenGrid     = 2**cntrSize
    for mode in modes:
        if mode.startswith('FP'):
            expSize = int(mode.split ('_e')[1])
            mode = 'FP'
            plotRecords.append ({
                'mode'  : mode,
                'label' : ResFileParser.genFpLabel(expSize=expSize, mantSize=cntrSize-expSize),
                'grid'  : scaleGrid (getAllValsFP(cntrSize=cntrSize, expSize=expSize, verbose=verbose, signed=signed)), 
                })
        elif mode.startswith('F2P'):
            flavor = mode.split('_')[1]
            plotRecords.append ({
                'mode'  : mode,
                'label' : ResFileParser.genF2pLabel(flavor=flavor),
                'grid'  : scaleGrid (getAllValsF2P (flavor=flavor, cntrSize=cntrSize, hyperSize=hyperSize, verbose=verbose, signed=signed)) 
                })

    for i in range(len(plotRecords)): 
        plotRecord = plotRecords[i]     
        ax.plot (plotRecord['grid'], 
                 [len(plotRecords)-i for item in range(lenGrid)], 
                 color      = colors[i], 
                 marker     = markerOfMode[mode], 
                 linestyle  = 'None', 
                 markersize = MARKER_SIZE_SMALL, 
                 label      = plotRecord['label'])  # Plot the conf' interval line
    frame1 = plt.gca()
    frame1.axes.get_yaxis().set_visible(False)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend (by_label.values(), by_label.keys(), fontsize=LEGEND_FONT_SIZE, frameon=False)        
    plt.show()

def quantizeWoRnd (vec : np.array, grid : np.array) -> np.array:
    """
    Quantize an input vector, using symmetric Min-max quantization. This is done by scaling the vector.
    The function does NOT round the vector.
    """
    upperBnd = max (abs(vec[0]), abs(vec[-1])) # The upper bound is the largest absolute value in the vector to quantize.
    lowerBnd = -upperBnd
    vec = clamp (vec, lowerBnd, upperBnd)
    scale = (upperBnd-lowerBnd) / (grid[-1]-grid[0])
    return [item/scale for item in vec] 
    
def calcAbsQuantErrorSortedVecs (grid, vec):
    """
    Given a sorted grid and a sorted quantized vec, return the quantizatoin error for each item in the quantized vec.
    """
    vecOfErrs = np.empty (len(vec))
    idxInGrid = 0
    for idxInVec in range(len(vec)):
        if idxInGrid==len(grid): # already reached the max grid val --> all next items in vec should be compared to the last item in the grid 
            vecOfErrs[idxInVec] = abs (vec[idxInVec]-grid[-1])
            continue
        vecOfErrs[idxInVec] = abs (vec[idxInVec]-grid[idxInGrid])
        while (idxInGrid < len(grid)):
            absErr = abs (vec[idxInVec]-grid[idxInGrid])
            if absErr <= vecOfErrs[idxInVec]:
                vecOfErrs[idxInVec] = absErr
                idxInGrid += 1
            else:
               idxInGrid -= 1
               break 
    return vecOfErrs

def calcMseSortedVecs (grid, vec):
    """
    Calculate the Mean Square Error between the grid and the (quantized) vector 
    """
    overallAbsErr = 0
    overallRelErr = 0
    idxInGrid = 0
    for idxInVec in range(len(vec)):
        if idxInGrid==len(grid): # already reached the max grid val --> all next items in vec should be compared to the last item in the grid 
            sqAbsErr        = (vec[idxInVec]-grid[-1])**2
            overallAbsErr += sqAbsErr
            overallRelErr += sqAbsErr/(vec[idxInVec]**2)
            continue
        curAbsErr = abs (vec[idxInVec]-grid[idxInGrid])
        while (idxInGrid < len(grid)):
            absErr = abs (vec[idxInVec]-grid[idxInGrid])
            if absErr <= curAbsErr:
                curAbsErr = absErr
                idxInGrid += 1
            else:
               idxInGrid -= 1
               break
        sqAbsErr        = curAbsErr**2
        overallAbsErr  += sqAbsErr
        overallRelErr  += sqAbsErr/(vec[idxInVec]**2)
    return {'abs' : overallAbsErr/len(vec), 'rel' : overallRelErr/len(vec)} 

def genVec2Quantize (dist     : 'uniform',  # distribution from which points are drawn  
                     lowerBnd : float,      # lower bound for the generated points  
                     upperBnd : float,      # upper bound for the generated points
                     numPts : int           # Num of points in the generated vector
                     ) -> np.array:
    """
    Generate a vector to be quantized.
    """
    if dist=='uniform':
        return np.array([(lowerBnd + i*(upperBnd-lowerBnd)/numPts) for i in range(numPts)])
    elif dist=='Gaussian':
        return (np.sort (np.random.randn(numPts) * (upperBnd/2)))
    else:
        settings.error ('In Quantization.py. Sorry. The distribution {dist} you chose is not supported.')
    
    
def simQuantErr (modes      = [], # modes to be simulated, e.g. FP, F2P_sr. 
                 cntrSize   = 8,  # of bits, including the sign bit 
                 expSizes   = [1], # size of the exponent when simulating FP 
                 hyperSize  = 2,  # size of the hyper-exp, when simulating F2P  
                 verbose    = []  # level of verbose, as defined in settings.py. 
                 ):
    """
    Simulate the required configuration and output the results (the quantization errors) as defined by the verbose.
    """
    
    np.random.seed (settings.SEED)
    vec2quantize = genVec2Quantize (dist='Gaussian', lowerBnd=-0.5, upperBnd=1, numPts = 100)
    for mode in modes:
        if mode=='FP':
            for expSize in expSizes: 
                grid     = getAllValsFP(cntrSize=cntrSize, expSize=expSize, verbose=verbose, signed=True)                
                print (f'grid[0]={grid[0]}, grid[-1]={grid[-1]}')
                MSE = calcMseSortedVecs (grid=grid, vec=quantizeWoRnd (vec=vec2quantize, grid=grid))
                print ('{}, abs_MSE={}, rel_MSE={}' .format(ResFileParser.genFpLabel(expSize=expSize, mantSize=cntrSize-expSize), MSE['abs'], MSE['rel']))
        elif mode.startswith('F2P'):
            flavor = mode.split('_')[1]
            grid = getAllValsF2P (flavor=flavor, cntrSize=cntrSize, hyperSize=hyperSize, verbose=verbose, signed=True)
            MSE = calcMseSortedVecs (grid=grid, vec=quantizeWoRnd (vec=vec2quantize, grid=grid))
            print ('{}, abs_MSE={}, rel_MSE={}' .format(ResFileParser.genF2pLabel(flavor=flavor), MSE['abs'], MSE['rel']))
        else:
            settings.error ('Sorry, the requested mode {mode} is not supported.')
# simQuantErr (modes=['F2P_sr', 'FP'], expSizes=[1,6]) #'F2P_sr', 
plotScaledGrids (cntrSize=6, modes=['FP_e1', 'F2P_sr', 'FP_e5', 'F2P_lr'])
