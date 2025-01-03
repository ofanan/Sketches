import os, time, scipy, matplotlib, pickle, numpy as np, seaborn as sns
# from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from fitter import Fitter, get_common_distributions, get_distributions

import settings, ResFileParser, F2P_sr, F2P_lr, F2P_li, FP, SEAD_stat, SEAD_dyn  
from tictoc import tic, toc
from printf import printf, printar, printarFp
from SingleCntrSimulator import *
from ResFileParser import *
from settings import *

np.set_printoptions(precision=NP_PRINT_PRECISION) 
MAX_DF = 20

# Dequantize the given vector, namely, multiply each element in it by the given scale.
# Inputs: vec : np.array; vector to dequantize 
#         scale : float; scale factor5
#         z     : float; the zero-point
# Output: np.array; the dequantized vector
dequantize = lambda vec, scale, z : (vec-z)*scale 

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

def setPltParams (size : str = 'large') -> None:
    """
    Set the plot parameters (sizes, colors etc.).
    """
    matplotlib.rcParams.update({
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
    'axes.titlesize'    : FONT_SIZE_SMALL
    })


def clamp (
        vec      : np.array, 
        lowerBnd : float, 
        upperBnd : float) -> np.array:
    """
    Clamp a the input vector vec, as follows.
    For each item in vec:
    - if x<min(grid), assign x=lowrBnd
    - if x>max(grid), assign x=upperBnd
    """
    vec[vec < lowerBnd] = lowerBnd
    vec[vec > upperBnd] = upperBnd
    return vec 

def calcErr (
        orgVec         : np.array, # vector before quantization 
        changedVec     : np.array, # vector after quantization+dequantization
        weightDist     : str = None, # distribution by which the MSE is weighted; when None, do not calculate the weighted MSE
        stdev          : float = None,       # standard variation of the distribution; the expected value is 0.
        scale          : float = None,       # the scale by which orgVec was quantized
        logFile        = None, # object for the logFile; to be used if the verbose requests for logFile
        recordErrVecs  = False, # When True, add the error vector (and not only their means) to the returned resRecord.
        verbose        : list = []    # level of verbose, as defined in settings.py 
    ):
    """
    Calculate the errors between the original vector and the changed vector.
    The errors consider are:
    - absolute/relative.
    - Regular - MSE (Mean Square Error).
    - The Mse, weighted by the given distribution and stdev (standard variation). 
    """
    absErrVec = np.abs(orgVec-changedVec) 
    absSqErrVec = np.square(absErrVec)
    relErrVec = [absErrVec[i]/orgVec[i] for i in range(len(orgVec)) if orgVec[i]!=0]
    relSqErrVec = np.square(relErrVec)
    resRecord = {
            'scale'  : scale, 
            'abs'    : np.mean (absErrVec),
            'absMse' : np.mean (absSqErrVec), 
            'relMse' : np.mean (relSqErrVec) 
        } 
    if recordErrVecs:
        resRecord['absErrVec'] = absErrVec
    if weightDist==None: # no need to calculate weighted Mse
        return resRecord

    if weightDist!='norm':
        error (f'In FPQuantization.calcErr(). Sorry, the distribution {dist} you chose is not supported.')
    if stdev==None:
        error (f'In FPQuantization.calcErr(). Please specify stdev.')        
    pdfVec = [scipy.stats.norm(0, stdev).pdf(orgVec[i]) for i in range(len(orgVec))]
    weightedAbsMseVec      = np.dot (pdfVec, absSqErrVec) 
    weightedRelMseVec      = np.empty((len(orgVec[orgVec!=0])))
    idxInweightedRelMseVec = 0
    for i in range(len(orgVec)):
        if orgVec[i]==0:
            continue
        weightedRelMseVec[idxInweightedRelMseVec] = scipy.stats.norm(0, stdev).pdf(orgVec[i])*((orgVec[i]-changedVec[i])/orgVec[i])**2 
        idxInweightedRelMseVec += 1

    if VERBOSE_LOG in verbose:
        printf (logFile, f'// mode={mode}\n')
        for i in range (10):
             printf (logFile, f'i={i}, org={orgVec[i]}, changed={changedVec[i]}, PDF={scipy.stats.norm(0, stdev).pdf(orgVec[i])}, weightedAbsMse={weightedAbsMseVec[i]}\n')
    
    resRecord['avgWeightedAbsMse'] = np.mean (weightedAbsMseVec)
    resRecord['avgWeightedRelMse'] = np.mean (weightedRelMseVec)
    if recordErrVecs:
        resRecord['weightedAbsMseVec'] = weightedAbsMseVec
        resRecord['weightedRelMseVec'] = weightedRelMseVec
    return resRecord

def scaleGrid (grid : np.array, lowerBnd=0, upperBnd=100) -> np.array:
    """
    Scale the given sorted grid into the given range [lowerBnd, upperBnd]
    """
    error ('Please check the new, np version, of this function')
    scale = (upperBnd-lowerBnd) / (grid[-1]-grid[0])
    return scale * grid  
    # return [item*scale for item in grid] 
    
def quantize (
    vec                : np.array, # The vector to quantize 
    grid               : np.array,  # The quantization grid (all the values that can be represented by the destination number representation
    clampOutliers      : bool = False, # When True, clamp vec by removing all the values outside [lowerBnd, upperBnd], to remove outliers.
    lowerBnd           : float = None,
    upperBnd           : float = None,
    useAsymmetricQuant : bool = False, # When True, use asymmetric quantization
    verbose            : list = [], # verobse method, defined at settings.py
    debugFile           = None
    ) -> [np.array, float]: # [the_quantized_vector, the scale_factor (by which the vector was divided)] 
    """
    Quantize an input vector, using Min-max quantization. 
    This is done by:
    - Quantizing the vector, namely:
      - Clamping (if requested).
      - Scaling the vector. The scaling method is minMax.
      - If the quantization is asymmetric, calculate an offest, add add it to the vector (not supproted yet). 
      - Rounding the vector to the nearest values in the grid.
    """
    grid = np.array (grid)
    vec  = np.array (vec)
    if clampOutliers:
        if lowerBnd==None or upperBnd==None:
            error ('In Quantizer.quantize(). Clamp where requested, but lowerBnd or upperBnd was not specified.') 
        vec   = clamp (vec, lowerBnd, upperBnd)
    scale       = (max(vec)-min(vec)) / (max(grid)-min(grid))
    if useAsymmetricQuant:
        if min(grid)!=0:
            error ('In Quantizeer.quant(). Asymmetric quantization is supported only for unsigned grid.')
        error ('In Quantizeer.quant(). Sorry, but asymmetric quantization is currently not supported')
    else:
        z = 0
    scaledVec   = vec/scale + z# The vector after scaling and clamping (still w/o rounding)  
    if str(grid.dtype).startswith('int'):
        return [scaledVec.astype('int'), scale, z]
    grid        = np.sort (grid)

    if VERBOSE_DEBUG in verbose:
        printf (debugFile, f'scale={scale}\nscaledVec={scaledVec}\n')
    sorted_indices    = np.argsort(scaledVec) # Get the indices that would sort the array
    undo_sort_indices = np.argsort(sorted_indices)
    scaledVec      = scaledVec[sorted_indices] # sort sclaedVec
    if VERBOSE_DEBUG in verbose:
        printf (debugFile, f'sorted_indices={sorted_indices}\nsorted scaledVec={scaledVec}\n')
    quantVec        = np.empty (len(vec)) # The quantized vector (after rounding scaledVec) 
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
               quantVec[idxInVec] = grid[idxInGrid]
               break
    if VERBOSE_DEBUG in verbose:
        printf (debugFile, f'undo_sort_indices={undo_sort_indices}\nsorted quantized vec={quantVec}\n')
    return [quantVec[undo_sort_indices], scale, z]

def genRandVec2Quantize (
        dist         : str   = 'uniform',  # distribution from which points are drawn  
        lowerBnd     : float = 0,   # lower bound for the generated points  
        upperBnd     : float = 10,   # upper bound for the generated points
        stdev        : float = 1,   # standard variation when generating a Gaussian dist' points
        vec2quantLen : int   = 1000, # Num of points in the generated vector
        outLier      : float = None,
    ) -> np.array:
    """
    Generate an np.array to be quantized, using the requested distribution.
    """
    if dist=='uniform':
        vec = [(lowerBnd + i*(upperBnd-lowerBnd)/(vec2quantLen-1)) for i in range(vec2quantLen)] #$$$ change to np-style to boost perf'
    elif dist=='norm':
        rng = np.random.default_rng(SEED)
        vec = np.sort (rng.standard_normal(vec2quantLen) * stdev)
    elif dist.startswith('t_'):
        vec = np.sort (np.random.standard_t(df=getDf(dist), size=vec2quantLen) * stdev)
    elif dist=='int': # vector of integers in the range
        vec = np.arange (lowerBnd, upperBnd+1) 
    else:
        error (f'In Quantization.genRandVec2Quantize(). Sorry. The distribution {dist} you chose is not supported.')
    if outLier==None:
        return np.array (vec)
    return np.array ([-outLier] + vec + [outLier])
    
def calcQuantRoundErr (
        cntrSize       : int   = 8,  # of bits, including the sign bit
        modes          : list  = [], # modes to be simulated, e.g. FP, F2P_sr. 
        signed         : bool  = False, # When True, consider a signed counter
        vec2quantize   : np.array  = None, # The vector quantize.
        inputFrom      : str = None, # A string that details the origin of the vector to quantize  
        verbose        : list  = [],  # level of verbose, as defined in settings.py.
    ):
    """
    Simulate the required configurations, and calculate the rounding quantization errors. Output the results (the quantization rounding errors) as defined by the verbose.
    """
    np.random.seed (SEED)
    if VERBOSE_RES in verbose:
        resFile = open (f'../res/{genRndErrFileName(cntrSize)}.res', 'a+')
    if VERBOSE_LOG in verbose:
        logFile = open (f'../res/quant_n{cntrSize}.log', 'w')
    else:        
        logFile = None

    if VERBOSE_PCL in verbose:
        outputFileName = genRndErrFileName (cntrSize)
        pclOutputFile = open(f'../res/pcl_files/{outputFileName}.pcl', 'ab+')
    
    resRecords = []
    for mode in modes:
        if VERBOSE_DEBUG in verbose:
            debugFile = open ('../res/debug.txt', 'a+')
            printf (debugFile, f'// mode={mode}\n')
        if mode.startswith('FP'):
            expSize = int(mode.split ('_e')[1])
            grid                     = getAllValsFP(cntrSize=cntrSize, expSize=expSize, verbose=[], signed=signed)
                
        elif (mode.startswith('F2P') or mode.startswith('F3P')):
            grid = getAllValsFxp (
                fxpSettingStr = mode,
                cntrSize    = cntrSize, 
                verbose     = [], 
                signed      = signed
            )
            
        elif mode.startswith('int'):
            if signed: 
                grid = np.array (range(-2**(cntrSize-1)+1, 2**(cntrSize-1), 1), dtype='int')
            else:
                grid = np.array (range(2**cntrSize), dtype='int')
        
        elif mode.startswith ('SEAD_stat'):
            expSize = int(mode.split('_e')[1].split('_')[0])
            myCntrMaster = SEAD_stat.CntrMaster (cntrSize=cntrSize, expSize=expSize, verbose=verbose)            
            grid = myCntrMaster.getAllVals ()

        elif mode.startswith ('SEAD_dyn'):
            myCntrMaster = SEAD_dyn.CntrMaster (cntrSize=cntrSize, verbose=verbose)            
            grid = myCntrMaster.getAllVals ()

        else:
            print (f'In Quantizer.calcQuantRoundErr(). Sorry, the requested mode {mode} is not supported.')
            continue

        [quantizedVec, scale, z] = quantize(vec=vec2quantize, grid=grid)
        dequantizedVec           = dequantize(vec=quantizedVec, scale=scale, z=z)
        
        # Analyze the results of this experiment and insert them into resRecord 
        resRecord = calcErr( 
            orgVec      = vec2quantize, 
            changedVec  = dequantizedVec, 
            scale       = scale,
            logFile     = logFile,
            verbose     = verbose
        )
        resRecord['mode']       = mode
        resRecord['signed']  = signed
        resRecord['numPts']     = len (vec2quantize)
        resRecord['inputFrom']  = inputFrom
        
        if VERBOSE_DEBUG in verbose:
            debugFile = open ('../res/debug.txt', 'a+')
            for i in range(len(vec2quantize)):
                printf (debugFile, f'i={i}, vec[i]={vec2quantize[i]}, quantizedVec[i]={quantizedVec[i]}, dequantizedVec={dequantizedVec[i]}\n')
            printf (debugFile, '\n')
            exit ()

        if VERBOSE_COUT_CNTRLINE in verbose:
            print (resRecord)
        
        if VERBOSE_RES in verbose:
            printf (resFile, f'{resRecord}\n')
            # for key, value in resRecord.items():
            #     if not key.endswith('Vec'):
            #         printf (resFile, f'{key} : {value}\n')
            # printf (resFile, '\n')
        
        if VERBOSE_PCL in verbose:
            pickle.dump(resRecord, pclOutputFile)        
        if VERBOSE_PLOT in verbose:
            resRecords.append (resRecord)

def plotGrids (
        modes       = [], # modes to be simulated, e.g. FP, F2P_sr. 
        cntrSize    = 7,  # of bits, including the sign bit 
        hyperSize   = 2,  # size of the hyper-exp, when simulating F2P  
        signed      = False, # when True, plot also negative values (symmetrically w.r.t. 0).
        zoomXlim    = None,  # when not None, generate the plot zoomed so that x values are up to this value
        scale       = False, # When True, scale the grid by the int's range.  
        verbose     = []  # level of verbose, as defined in settings.py.
        ) -> None:
    """
    """
    setPltParams ()
    _, ax       = plt.subplots()
    resRecords  = []
    lenGrid     = 2**cntrSize
    
    # Set lowerBnd and upperBnd of the range to be plotted
    if scale:
        lowerBnd    = 0
        upperBnd = 2**cntrSize-1 # scale by int range
    else: # Do not scale the grids; lowerBnd, upperBnd will be assigned the smallest, largest value to be plotted
        lowerBnd = 1
        upperBnd = 0 
    for mode in modes:
        if mode.startswith('FP'):
            expSize = int(mode.split ('_e')[1])
            grid = getAllValsFP(cntrSize=cntrSize, expSize=expSize, verbose=verbose, signed=signed)
            if scale:
                grid = scaleGrid (grid, lowerBnd = lowerBnd, upperBnd = upperBnd)
            else:
                print (f'in FP: grid[1]={grid[1]}')
                lowerBnd = min (lowerBnd, grid[1])
                upperBnd = max (upperBnd, grid[-1])
            resRecord = {
                'mode'  : mode,
                'grid'  : grid
                }
        elif mode.startswith('F2P'):
            numSettings = getFxpSettings (mode)
            flavor    = numSettings['flavor'] 
            grid    = getAllValsFxp (flavor=flavor, cntrSize=cntrSize, hyperSize=numSettings['hyperSize'], verbose=verbose, signed=signed)
            if scale:
                grid = scaleGrid (grid, lowerBnd = lowerBnd, upperBnd = upperBnd)
            else:
                print (f'in {mode}: grid[1]={grid[1]}')
                lowerBnd = min (lowerBnd, grid[1])
                upperBnd = max (upperBnd, grid[-1])
            resRecord = {
                'mode'  : mode,
                'grid'  : grid 
                }
        elif mode.startswith('int'):
            mode = 'FP'
            resRecord = {
                'mode'  : 'int',
                'grid'  : np.arange (0, 2**cntrSize) 
                }
            if not scale:
                upperBnd = max (upperBnd, grid[-1])
        else:
            settings.error (f'In Quantizer.plotGrids(). Sorry, the mode {mode} requested is not supported')
        resRecords.append (resRecord)
        
    legends=[]
    for i in range(len(resRecords)): 
        resRecord = resRecords[i]     
        curLine, = ax.plot (
            resRecord['grid'],
            np.array(range(lenGrid)), # Write the y index in reverse order, so that the legends' order will correspond the order of the plots. 
            color      = colorOfMode [resRecord['mode']], 
            linestyle  = 'None', 
            marker     = 'o',
            markersize = 1, 
            label      = labelOfMode(resRecord['mode'])
        )  # Plot the conf' interval line
        curLegend = ax.legend (handles=[curLine], bbox_to_anchor=(-0.24, i*(1.1/len(resRecords)), 0., .102), loc='lower left', frameon=False)
        ax.add_artist (curLegend)
    
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(True)
    if zoomXlim!=None:
        plt.xlim (0, zoomXlim)
    else:
        if (scale):
            plt.xlim (lowerBnd, upperBnd)
        else:
            plt.xlim (lowerBnd, upperBnd)
            plt.xscale ('log')               
    sns.despine(left=True, bottom=False, right=True)
    plt.savefig (f'../res/Grids_n{cntrSize}_I.pdf', bbox_inches='tight')

def npExperiments ():
    """
    Some experiments, to test np ops and speed.
    """
    rng = np.random.default_rng(settings.SEED)
    vecLen = 300
    orgVec = rng.random (vecLen)
    vec = np.arange (10.)
    mat = vec.reshape (10, 1)
    # print (f'mat={mat}')
    # print (f'shape={mat.shape}, type={mat.dtype}')
    print (vec[5:-5])

def testRandVecQuantRoundErr ():

    error ('In Quantizer.testRandVecQuantRoundErr(). Sorry, this function is not implemented yet.')
    verbose = [VERBOSE_PCL, VERBOSE_RES]
    stdev   = 1
    for cntrSize in np.array([8, 16, 19], dtype='uint8'):
        if VERBOSE_PCL in verbose:
            pclOutputFileName = f'{genRndErrFileName(cntrSize)}.pcl'
            # if os.path.exists(f'../res/pcl_files/{pclOutputFileName}'):
            #     os.remove(f'../res/pcl_files/{pclOutputFileName}')

        for distStr in ['uniform', 'norm', 't_5', 't_8']:
            ### To fully implement the func: 
            ### Call vec2quant = genRandVec2Quantize () 
            # Also, should open and write to .res file data about the distribution etc. 
            # if VERBOSE_RES in verbose:
            #     resFile = open (f'../res/{genRndErrFileName(cntrSize)}.res', 'a+')
            #     printf (resFile, f'// dist={dist}, stdev={stdev}, vec2quantLen={vec2quantLen}\n')
            #     if dist!='norm' and (not(dist.startswith('t_'))): 
            #         printf (resFile, f'// vecLowerBnd={vecLowerBnd}, vecUpperBnd={vecUpperBnd}, outLier={outLier}\n')
        # for distStr in ['uniform', 'norm', 't_5', 't_8', 't_2', 't_10', 't_4', 't_6', 't_8']:
            calcQuantRoundErr (
                cntrSize       = cntrSize, 
                modes          = settings.modesOfCntrSize(cntrSize), 
                verbose = verbose
            )   

def testQuantOfSingleVec (
        vec2quantize    : np.array,
        grid            : np.array,
        verbose         : list,
        debugFile       = None,
    ):
    """
    Test the quantization of a single vector and print the results as requested by verbose. 
    """
    if VERBOSE_DEBUG in verbose:
        printf (debugFile, f'vec2quantize={vec2quantize}\n')
    [quantizedVec, scale, z] = quantize (vec=vec2quantize, grid=grid, verbose=verbose, debugFile=debugFile) 
    dequantizedVec = dequantize (quantizedVec, scale, z) 
    if VERBOSE_PRINT_SCREEN in verbose:
        print (f'quantizedVec={quantizedVec}, scale={scale}, z={z}\ndequantizedVec={dequantizedVec}')
    if VERBOSE_DEBUG_DETAILS in verbose:
        printf (debugFile, f'grid={grid}\n')
    if VERBOSE_DEBUG in verbose:
        printf (debugFile, f'quantizedVec={quantizedVec}\nscale={scale}, z={z}\ndequantizedVec={dequantizedVec}')

def testQuantization (
        vecLen  : 5, # length of the vector to test
        verbose : list = [],
    ):
    """
    Basic test of the quantization
    """
    vec2quantize = 2 * np.random.rand(vecLen) - 1
    cntrSize = 8
    if VERBOSE_DEBUG in verbose or VERBOSE_DEBUG_DETAILS in verbose:
        debugFile = open ('../res/debug.txt', 'w')
    else:
        debugFile = None
     
    # grid = np.array (range(-2**(cntrSize-1)+1, 2**(cntrSize-1), 1), dtype='int')
    # testQuantOfSingleVec(vec2quantize=vec2quantize, grid=grid, verbose=verbose, debugFile=debugFile)
    fxpSettingStr = 'F3P_sr_h2'
    if VERBOSE_DEBUG in verbose or VERBOSE_DEBUG_DETAILS in verbose:
        printf (debugFile, f'// {fxpSettingStr}\n')
    grid = getAllValsFxp (
        fxpSettingStr = fxpSettingStr,
        cntrSize    = cntrSize, 
        verbose     = [], 
        signed      = True
    )
    testQuantOfSingleVec(vec2quantize=vec2quantize, grid=grid, verbose=verbose, debugFile=debugFile)
    if VERBOSE_DEBUG in verbose:
        debugFile.close ()
        
if __name__ == '__main__':
    try:
        testQuantization (verbose=[VERBOSE_DEBUG], vecLen=5)
        # runCalcQuantRoundErr ()
        # plotGrids (zoomXlim=None, cntrSize=7, modes=['F2P_li_h2', 'F2P_si_h2', 'FP_e5', 'FP_e2', 'int'], scale=False)

    except KeyboardInterrupt:
        print('Keyboard interrupt.')

# scaled 'F2P_lr_h1' is identical to int.