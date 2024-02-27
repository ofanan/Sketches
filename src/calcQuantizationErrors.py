"""
calculate the quantization error of a number representation system. 
"""
import os, math, pickle, time, random #sys
from printf import printf, printar, printarFp
import numpy as np #, scipy.stats as st, pandas as pd
import settings, SEAD_stat, CEDAR, Morris, AEE, F2P_sr, F2P_lr, F2P_li, FP  
from datetime import datetime

