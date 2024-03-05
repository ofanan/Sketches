"""
calculate the quantization error of a number representation system. 
"""

import os, math, pickle, time, random #sys
from printf import printf, printar, printarFp
import numpy as np #, scipy.stats as st, pandas as pd
import settings, SEAD_stat, CEDAR, Morris, AEE, F2P_sr, F2P_lr, F2P_li, FP  
from datetime import datetime

def calcQuantErrorElement (g, w):
    return min(g, key=lambda x: abs(w - x)) - w

def calcQuantErrorVec (grid, vec2quantize):
    return np.array([calcQuantErrorElement(grid, w) for w in vec2quantize])

vec2quantize = [-0.5, 4.5, 5, 10.5, 13]
grid         = [i for i in range (11)]
quantErVec = calcQuantErrorVec (vec2quantize=vec2quantize, grid=grid)
print (f'vec2quantize={vec2quantize}\ngrid={grid}\nquantErVec={quantErVec}')
