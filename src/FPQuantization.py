import numpy as np
import os
# from datetime import datetime
from tictoc import tic, toc
import matplotlib.pyplot as plt

from printf import printf, printar, printarFp
import settings, ResFileParser, F2P_sr, F2P_lr, F2P_li, FP  
from SingleCntrSimulator import main, getAllValsFP, getAllValsF2P

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

def calcAbsQuantErrorSortedVecs (grid, vec2quantize):
    vecOfErrs = np.empty (len(vec2quantize))
    idxInGrid = 0
    for idxInVec in range(len(vec2quantize)):
        if idxInGrid==len(grid): # already reached the max grid val --> all next items in vec should be compared to the last item in the grid 
            vecOfErrs[idxInVec] = abs (vec2quantize[idxInVec]-grid[-1])
            continue
        vecOfErrs[idxInVec] = abs (vec2quantize[idxInVec]-grid[idxInGrid])
        while (idxInGrid < len(grid)):
            absErr = abs (vec2quantize[idxInVec]-grid[idxInGrid])
            if absErr <= vecOfErrs[idxInVec]:
                vecOfErrs[idxInVec] = absErr
                idxInGrid += 1
            else:
               idxInGrid -= 1
               break 
    return vecOfErrs

def calcMseSortedVecs (grid, vec2quantize):
    overallAbsErr = 0
    idxInGrid = 0
    for idxInVec in range(len(vec2quantize)):
        if idxInGrid==len(grid): # already reached the max grid val --> all next items in vec should be compared to the last item in the grid 
            overallAbsErr += (vec2quantize[idxInVec]-grid[-1])**2
            continue
        curAbsErr = abs (vec2quantize[idxInVec]-grid[idxInGrid])
        while (idxInGrid < len(grid)):
            absErr = abs (vec2quantize[idxInVec]-grid[idxInGrid])
            if absErr <= curAbsErr:
                curAbsErr = absErr
                idxInGrid += 1
            else:
               idxInGrid -= 1
               break
        overallAbsErr += curAbsErr**2
    return (overallAbsErr/len(vec2quantize))

# def symmetricQuant (vec2quantize, grid):
#     """
#     Perform symmetric quantization. vec2quantize and 
#     Each item x in vec2quantize is quantized as follows:
#     - clamp x to a value within the grid:
#         - if x<min(grid), assign x=min(grid)
#         - if x>max(grid), assign x=max(grid)
#     - set x to the closest value within the grid. This is equivalent to setting x=round(x/scale). (To be checked).
#     """
#     vec2quantize = clamp (vec2quantize)
    
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
    else:
        settings.error ('In Quantization.py. Sorry. The distribution {dist} you chose is not supported.')
    
    
def simQuantErr (modes=[], cntrSize=8, expSizes=[], hyperSize=2, verbose=[]):
    
    vec2quantize = genVec2Quantize (dist='uniform', lowerBnd=-100, upperBnd=100, numPts = 1000)
    cntrSize = cntrSize-1 # account for the sign bit
    for mode in modes:
        if mode=='FP':
            for expSize in expSizes: 
                grid     = getAllValsFP(cntrSize=cntrSize, expSize=expSize, signed=False, verbose=verbose)                
                clampedVec2quantize = clamp (vec=vec2quantize, lowerBnd=grid[0], upperBnd=grid[-1]) # getAllVals returns the grid sorted, so the smallest, largest values are the first, last ones
                MSE = calcMseSortedVecs(grid=grid, vec2quantize=clampedVec2quantize)
                print (f'{ResFileParser.genFpLabel(expSize=expSize, mantSize=cntrSize-expSize)}, MSE={MSE}')
        elif mode.startswith('F2P'):
            flavor = mode.split('_')[1]
            grid = getAllValsF2P (flavor=flavor, cntrSize=cntrSize, hyperSize=hyperSize, verbose=verbose)                
            clampedVec2quantize = clamp (vec=vec2quantize, lowerBnd=grid[0], upperBnd=grid[-1]) # getAllVals returns the grid sorted, so the smallest, largest values are the first, last ones
            MSE = calcMseSortedVecs(grid=grid, vec2quantize=clampedVec2quantize)
            print (f'{ResFileParser.genF2pLabel(flavor=flavor)}, MSE={MSE}')
        else:
            settings.error ('Sorry, the requested mode {mode} is not supported.')
simQuantErr (modes=['F2P_sr', 'FP'], expSizes=[1,6])