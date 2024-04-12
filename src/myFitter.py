import numpy as np
import pandas as pd
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
from collections import Counter

import settings

MAX_DF = 30

def myFitter (
        vec : np.array,
        ) -> str:
    """
    Find the distribution that best fits the given vector.
    If all fit tests agree, return a string that represents the distribution they all agree on.
    Else, return None
    """
    f = Fitter (vec, 
                distributions = ['t', 'uniform', 'norm'] # distributions to consider
                )
    f.fit ()

    likelihoodTests = ['sumsquare_error', 'bic', 'ks_statistic']
    suggestedDists  = [None]*len(likelihoodTests) # suggestedDists[i] will get the best-fit distributions accordingy to test i 
    for i in range(len(likelihoodTests)):   
        for distByThisTest in f.get_best(likelihoodTests[i]):
            suggestedDists[i] = distByThisTest
    c = Counter (suggestedDists)
    dist, numTests = c.most_common(1)[0]
    if numTests==len(likelihoodTests): # all tests agree
        distDict = f.get_best(likelihoodTests[0])
        for distName in distDict:
            if distName!='t': # For distributions other than Student-t, no need additional parameters
                return distName
            # Now we know that the distribution found is 't'. 
            df = distDict['t']['df']
            if df > MAX_DF:
                return 'norm'
            return f't_{df}'
    else:
        return None

# vec=np.random.standard_t(df=8, size=1000)
# rng = np.random.default_rng(settings.SEED)
# vec = np.sort (rng.standard_normal(1000))
# print (findDist (vec))
