# Run functions analyzing an F2P counter, e.g.: print all the possible values, or the maximal values, at a concrete configuration. 
import math, random, pickle
from printf import printf
import settings
import numpy as np

import F2P_lr, F2P_sr, F2P_li



printAllVals (cntrSize=5, hyperSize=1, verbose=[settings.VERBOSE_RES, settings.VERBOSE_COUT_CONF, settings.VERBOSE_COUT_CNTRLINE], flavor='li') #, settings.VERBOSE_COUT_CNTRLINE
# printAllCntrMaxVals (hyperSizeRange=[1,2], cntrSizeRange=[6,7,8,9,10,11,12,13,14,15,16], verbose=[settings.VERBOSE_RES], flavor='li')
# coutConfData (cntrSize=6, hyperSize=1, flavor='li')
