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

def calcQuantErrorVec (grid, vec2quantize):
    return np.array([calcQuantErrorElement(grid, v) for v in vec2quantize])

def calcQuantErrorVecMy (grid, vec2quantize):
    vecOfErrs = np.empty (len(vec2quantize))
    idxInGrid = 0
    for idxInVec in range(len(vec2quantize)):
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
        
    return vecOfErrs

lenOfGrid    = 2**7
lenOfVec     = 10000
vec2quantize = [(-7 + 0.01*i) for i in range(lenOfVec)]
grid         = [i for i in range (lenOfGrid+1)]
tic ()
quantErVec = calcQuantErrorVec (vec2quantize=vec2quantize, grid=grid)
toc ()
tic ()
quantErVecMy = calcQuantErrorVecMy (vec2quantize=vec2quantize, grid=grid)
toc ()
# print (f'quantErVec={quantErVec}\nquantErVecMy={quantErVecMy}\ndiff={quantErVec-quantErVecMy}')
sumDiff = sum(abs(quantErVec-quantErVecMy))
print (f'sumDiff={sumDiff}')