"""
calculate the quantization error of a number representation system. 
"""
import os, math, pickle, time, random #sys
from printf import printf, printar, printarFp
import numpy as np #, scipy.stats as st, pandas as pd
import settings, SEAD_stat, CEDAR, Morris, AEE, F2P_sr, F2P_lr, F2P_li, FP  
from datetime import datetime

class QuantErrorCalculator (object):
    """
    """
    
    def __init__ (self):
        self.weights = [2*i for i in range (6)]
    
    def calcQuantErrorElement (self, g, w):
        return min(g, key=lambda x: abs(w - x)) - w
    
    def calcQuantErrorVec (self, grid):
        self.Rvec = np.array([self.calcQuantErrorElement(grid, w) for w in self.weights])
        print (f'grid={grid}, weights={self.weights}, R_vec={self.Rvec}')



myQErrorCalc = QuantErrorCalculator()
grid = [7, 13]
myQErrorCalc.calcQuantErrorVec (grid=grid)

