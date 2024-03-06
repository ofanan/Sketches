"""
calculate the quantization error of a number representation system. 
"""

import os, math, pickle, time, random #sys
from printf import printf, printar, printarFp
import numpy as np #, scipy.stats as st, pandas as pd
import settings, SEAD_stat, CEDAR, Morris, AEE, F2P_sr, F2P_lr, F2P_li, FP  
from datetime import datetime
from tictoc import tic, toc

def calcQuantErrorElement (g, v):
    return min(g, key=lambda x: abs(v - x)) - v

def calcAbsQuantErrorVec (grid, vec2quantize):
    return abs(np.array([calcQuantErrorElement(grid, v) for v in vec2quantize]))

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

large = True
if (large):
    lenOfGrid    = 2**7
    lenOfVec     = 50000
    vec2quantize = [(-0.05*lenOfVec + 0.1*i) for i in range(lenOfVec)]
    grid         = [i for i in range (lenOfGrid)]
else:
    vec2quantize = [-0.5, 4.5, 5.5, 10.5]
    grid         = [0, 5, 10]
tic ()
quantErVec = calcAbsQuantErrorVec (vec2quantize=vec2quantize, grid=grid)
# Mse = np.mean([item**2 for item in quantErVec])
# if (large):
#     toc ()
tic ()
MseMy = calcMseSortedVecs(grid=grid, vec2quantize=vec2quantize)
if (large):
    toc ()
# MseDiff = MseMy - Mse
# print (f'MseDiff={MseDiff}, rel_err={MseDiff/MseMy}')

