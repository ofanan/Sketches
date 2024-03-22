import os, scipy, seaborn, matplotlib, pickle, numpy as np
# from datetime import datetime
import matplotlib.pyplot as plt

import settings, ResFileParser, F2P_sr, F2P_lr, F2P_li, FP  
from tictoc import tic, toc
from printf import printf, printar, printarFp
from SingleCntrSimulator import main, getAllValsFP, getAllValsF2P
from ResFileParser import getF2PSettings, colors, colorOfLabel, markerOfMode, MARKER_SIZE_SMALL, FONT_SIZE, LEGEND_FONT_SIZE

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

def dequantize (vec : np.array, scale : float) -> np.array:
    """
    Dequantize the given vector, namely, multiply each element in it by the given scale.
    """
    return [item*scale for item in vec] 

def calcMse (orgVec     : np.array, # vector before quantization 
             changedVec : np.array, # vector after quantization+dequantization
             weightDist : str = None, # distribution by which the MSE is weighted; when None, do not calculate the weighted MSE
             stdev      : float = 0.01,       # standard variation of the distribution; the expected value is 0.
             scale      : float = None,       # the scale by which orgVec was quantized
             logFile     = None, # object for the logFile; to be used if the verbose requests for logFile
             recordErrVec = False, # When True, add absErrVec to the returned resRecord.
             verbose    : list = []    # level of verbose, as defined in settings.py 
             ):
    """
    Calculate the: 
    - MSE (Mean Square Error), both relative and absolute, between the original vector and the changed vector.
    - The Mse, weighted by the given distribution and stdev (standard variation). 
    """
    absErrVec = [abs(orgVec[i]-changedVec[i]) for i in range(len(orgVec))]
    resRecord = {
            'scale'  : scale, 
            'relMse' : np.mean ([((orgVec[i]-changedVec[i])/orgVec[i])**2 for i in range(len(orgVec)) if orgVec[i]!=0]),
            'absMse' : np.mean (absErrVec),
        } 
    
    if recordErrVec:
        resRecord['absErrVec'] = absErrVec
    if weightDist==None: # no need to calculate weighted Mse
        return resRecord

    if weightDist!='Gaussian':
        settings.error (f'In FPQuantization.calcMse(). Sorry, the distribution {dist} you chose is not supported.')
    weightedAbsMseVec      = [scipy.stats.norm(0, stdev).pdf(orgVec[i])*(orgVec[i]-changedVec[i])**2 for i in range(len(orgVec))]
    weightedRelMseVec      = np.empty(len([item for item in orgVec if item!=0]))
    idxInweightedRelMseVec = 0
    for i in range(len(orgVec)):
        if orgVec[i]==0:
            continue
        weightedRelMseVec[idxInweightedRelMseVec] = scipy.stats.norm(0, stdev).pdf(orgVec[i])*((orgVec[i]-changedVec[i])/orgVec[i])**2 
        idxInweightedRelMseVec += 1

    if settings.VERBOSE_LOG in verbose:
        printf (logFile, f'// Label={label}\n')
        for i in range (10):
             printf (logFile, f'i={i}, org={orgVec[i]}, changed={changedVec[i]}, PDF={scipy.stats.norm(0, stdev).pdf(orgVec[i])}, weightedAbsMse={weightedAbsMseVec[i]}\n')
    
    resRecord['weightedAbsMseVec'] = weightedAbsMseVec
    resRecord['avgWeightedAbsMse'] = np.mean (weightedAbsMseVec),
    resRecord['weightedRelMseVec'] = weightedRelMseVec,
    resRecord['avgWeightedRelMse'] = np.mean (weightedRelMseVec)
    return resRecord

def scaleGrid (grid : np.array, lowerBnd=0, upperBnd=100) -> np.array:
    """
    Scale the given sorted grid into the given range [lowerBnd, upperBnd]
    """
    scale = (upperBnd-lowerBnd) / (grid[-1]-grid[0])
    return [item*scale for item in grid] 
    
    
def quantize (vec  : np.array, # The vector to quantize 
              grid : np.array  # The quantization grid (all the values that can be represented by the destination number representation
              ) -> [np.array, float]: # [the_quantized_vector, the scale_factor (by which the vector was divided)] 
    """
    Quantize an input vector, using symmetric Min-max quantization. 
    This is done by:
    - Quantizing the vector, namely:
      - Clamping and scaling the vector. The scaling method is minMax.
      - Rounding the vector to the nearest values in the grid.
    """
    upperBnd    = max (abs(vec[0]), abs(vec[-1])) # The upper bound is the largest absolute value in the vector to quantize.
    lowerBnd    = -upperBnd
    scaledVec   = clamp (vec, lowerBnd, upperBnd)
    scale       = (upperBnd-lowerBnd) / (grid[-1]-grid[0])
    scaledVec   = [item/scale for item in vec] # The vector after scaling and clamping (still w/o rounding)  
    quantVec    = np.empty (len(vec)) # The quantized vector (after rounding scaledVec) 
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
    return [quantVec, scale]

def genVec2Quantize (dist       : str   = 'uniform',  # distribution from which points are drawn  
                     lowerBnd   : float = 0,   # lower bound for the generated points  
                     upperBnd   : float = 10,   # upper bound for the generated points
                     stdev      : float = 1,   # standard variation when generating a Gaussian dist' points
                     df         : float = 1,   # the df (degree of freedom) parameter; relevant only when using the t-student dist'.
                     numPts     : int   = 1000, # Num of points in the generated vector
                     outLier    : float = None,
                     ) -> np.array:
    """
    Generate a vector to be quantized, using the requested distribution.
    """
    rng = np.random.default_rng(settings.SEED)
    if dist=='Uniform':
        vec = [(lowerBnd + i*(upperBnd-lowerBnd)/numPts) for i in range(numPts)]
    elif dist=='Gaussian':
        vec = np.sort (rng.standard_normal(numPts) * stdev)
    elif dist=='Student':
        vec = np.sort (np.random.standard_t(df=df, size=numPts) * stdev)
    elif dist=='int': # vector of integers in the range
        vec = [i for i in range (lowerBnd, upperBnd+1)]
    else:
        settings.error ('In Quantization.genVec2Quantize(). Sorry. The distribution {dist} you chose is not supported.')
    if outLier==None:
        return np.array (vec)
    return np.array ([-outLier] + vec + [outLier])
    
def simQuantErr (modes          : list  = [], # modes to be simulated, e.g. FP, F2P_sr. 
                 cntrSize       : int   = 8,  # of bits, including the sign bit 
                 hyperSize      : int   = 1,  # size of the hyper-exp, when simulating F2P
                 dist           : str   = 'Gaussiam', # distribution of the points to simulate  
                 numPts         : int   = 1000, # num of points in the quantized vec
                 stdev          : float = 1,   # standard variation of the vector to quantize, when drawn from a Gaussian dist'
                 df             : float = 1,  # the df (degree of freedom) parameter; relevant only when using the t-student dist'.  
                 vecLowerBnd    : float = -float('inf'), # lower Bnd of the generated vector to quantize, if drawn from a uniform dist'  
                 vecUpperBnd    : float = float('inf'),   # upper Bnd of the generated vector to quantize, if drawn from a uniform dist'
                 outLier        : float = None, # Outlier value, to be added to the generated vector
                 verbose        : list  = [],  # level of verbose, as defined in settings.py.
                 ):
    """
    Simulate the required configuration and output the results (the quantization errors) as defined by the verbose.
    """
    np.random.seed (settings.SEED)
    if settings.VERBOSE_RES in verbose:
        resFile = open (f'../res/quant_n{cntrSize}.res', 'a+')
        printf (resFile, f'// dist={dist}, stdev={stdev}, numPts={numPts}\n')
        if dist=='Student':
            printf (resFile, f'// df={df}\n')
        if dist not in ['Gaussian', 'Student']: 
            printf (resFile, f'// vecLowerBnd={vecLowerBnd}, vecUpperBnd={vecUpperBnd}, outLier={outLier}\n')
    if settings.VERBOSE_LOG in verbose:
        logFile = open (f'../res/quant_n{cntrSize}.log', 'w')
    else:        
        logFile = None

    if settings.VERBOSE_PCL in verbose:

        pclOutputFileName = ResFileParser.genMsePclFileName (cntrSize)
        pclOutputFile = open(f'../res/pcl_files/{pclOutputFileName}', 'ab+')
    
    vec2quantize = genVec2Quantize (
        dist        = dist, 
        lowerBnd    = vecLowerBnd,   # lower bound for the generated points  
        upperBnd    = vecUpperBnd,   # upper bound for the generated points
        stdev       = stdev, 
        outLier     = outLier,
        df          = df,
        numPts      = numPts)
    if dist=='Uniform':
        weightDist = 'Gaussian'
    else:
        weightDist = None
    _, ax = plt.subplots()
    resRecords = []
    for mode in modes:
        if mode.startswith('FP'):
            expSize = int(mode.split ('_e')[1])
            mode = 'FP'
            grid                    = getAllValsFP(cntrSize=cntrSize, expSize=expSize, verbose=[], signed=True)
            [quantizedVec, scale]   = quantize(vec=vec2quantize, grid=grid)
            dequantizedVec          = dequantize(vec=quantizedVec, scale=scale)
            resRecord = calcMse(
                    orgVec      = vec2quantize, 
                    changedVec  = dequantizedVec, 
                    stdev       = stdev,
                    scale       = scale,
                    logFile     = logFile,
                    weightDist  = weightDist,
                    verbose     = verbose
                    )
            resRecord['label'] = ResFileParser.genFpLabel(expSize=expSize, mantSize=cntrSize-1-expSize)
                
        elif mode.startswith('F2P'):
            F2pSettings = getF2PSettings (mode)
            flavor    = F2pSettings['flavor']
            grid = getAllValsF2P (flavor=F2pSettings['flavor'], cntrSize=cntrSize, hyperSize=F2pSettings['hyperSize'], verbose=[], signed=True)
            [quantizedVec, scale] = quantize(vec=vec2quantize, grid=grid)                
            dequantizedVec = dequantize(vec=quantizedVec, scale=scale)
            resRecord = calcMse(
                    orgVec      = vec2quantize, 
                    changedVec  = dequantizedVec, 
                    scale       = scale,
                    stdev       = stdev,
                    logFile     = logFile,
                    weightDist  = weightDist,
                    verbose     = verbose
                    )
            resRecord['label'] = ResFileParser.f2pSettingsToLabel (mode)
            
        elif mode.startswith('int'):
            grid = np.array ([item for item in range (-2**(cntrSize-1)+1, 2**(cntrSize-1))])
            [quantizedVec, scale] = quantize(vec=vec2quantize, grid=grid)                
            dequantizedVec = dequantize(vec=quantizedVec, scale=scale)
            resRecord = calcMse(
                    orgVec      = vec2quantize, 
                    changedVec  = dequantizedVec, 
                    scale       = scale,
                    stdev       = stdev,
                    logFile     = logFile,
                    weightDist  = weightDist,
                    verbose     = verbose
                    )
            resRecord['label'] = 'int'
        elif mode=='shortTest':
            grid = np.array([i for i in range(-10, 11)])
            vec2quantize = np.array([-100, -95, -7, 99, 100])
            [quantizedVec, scale] = quantize(vec=vec2quantize, grid=grid)
            dequantizedVec = dequantize(vec=quantizedVec, scale=scale)
            resRecord = calcMse(
                    orgVec      = vec2quantize, 
                    changedVec  = dequantizedVec, 
                    )
            resRecord['label']  = 'shortTest'
        else:
            print (f'In Quantizer.simQuantErr(). Sorry, the requested mode {mode} is not supported.')
            continue

        if settings.VERBOSE_COUT_CNTRLINE in verbose:
            print (resRecord)
        
        if settings.VERBOSE_RES in verbose:
            for key, value in resRecord.items():
                if not key.endswith('Vec'):
                    printf (resFile, f'{key} : {value}\n')
            printf (resFile, '\n')
        
        resRecord['dist']   = dist
        resRecord['numPts'] = numPts
        resRecord['stdev']  = stdev
        if dist=='Student':
            resRecord['df'] = df
        if settings.VERBOSE_PCL in verbose:
            pickle.dump(resRecord, pclOutputFile)        
        if settings.VERBOSE_PLOT in verbose:
            resRecords.append (resRecord)
        
    if settings.VERBOSE_PLOT not in verbose:
        return
     
    for i in range(len(resRecords)): 
        resRecord = resRecords[i]
        ax.plot (vec2quantize, 
                 resRecord['weightedAbsMseVec'], 
                 color      = colorOfLabel[resRecord['label']], 
                # linestyle  = 'None', 
                 markersize = 2, 
                 label      = resRecord['label'])  # Plot the conf' interval line
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend (by_label.values(), by_label.keys(), fontsize=LEGEND_FONT_SIZE, frameon=False)
    # plt.yscale ('log')
    plt.xlim (-1, 1)
    plt.show()

def plotScaledGrids (
        modes       = [], # modes to be simulated, e.g. FP, F2P_sr. 
        cntrSize    = 7,  # of bits, including the sign bit 
        hyperSize   = 2,  # size of the hyper-exp, when simulating F2P  
        signed      = False, # when True, plot also negative values (symmetrically w.r.t. 0).
        zoomXlim    = None,  # when not None, generate the plot zoomed so that x values are up to this value  
        verbose     = []  # level of verbose, as defined in settings.py.
        ) -> None:
    """
    """
    setPltParams ()
    _, ax       = plt.subplots()
    resRecords  = []
    lenGrid     = 2**cntrSize
    lowerBnd    = 0
    upperBnd    = 2**cntrSize-1 
    for mode in modes:
        if mode.startswith('FP'):
            expSize = int(mode.split ('_e')[1])
            mode = 'FP'
            resRecord = {
                'mode'  : mode,
                'label' : ResFileParser.genFpLabel(expSize=expSize, mantSize=cntrSize-expSize),
                'grid'  : scaleGrid (getAllValsFP(cntrSize=cntrSize, expSize=expSize, verbose=verbose, signed=signed), lowerBnd = lowerBnd, upperBnd = upperBnd)
                }
        elif mode.startswith('F2P'):
            F2pSettings = getF2PSettings (mode)
            flavor    = F2pSettings['flavor'] 
            resRecord = {
                'mode'  : mode,
                'label' : ResFileParser.genF2pLabel(flavor=flavor, hyperSize=F2pSettings['hyperSize']),
                'grid'  : scaleGrid (getAllValsF2P (flavor=flavor, cntrSize=cntrSize, hyperSize=F2pSettings['hyperSize'], verbose=verbose, signed=signed), lowerBnd = lowerBnd, upperBnd = upperBnd) 
                }
        elif mode.startswith('int'):
            mode = 'FP'
            resRecord = {
                'mode'  : 'int',
                'label' : 'int',
                'grid'  : [i for i in range (lowerBnd, upperBnd+1)] 
                }
        else:
            settings.error (f'In Quantizer.plotScaledGrids(). Sorry, the mode {mode} requested is not supported')
        resRecords.append (resRecord)
        
    legends=[]
    for i in range(len(resRecords)): 
        resRecord = resRecords[i]     
        curLine, = ax.plot (resRecord['grid'], 
                 [i for item in range(lenGrid)], # len(resRecords)-i # Write the y index in reverse order, so that the legends' order will correspond the order of the plots. 
                 color      = colorOfLabel[resRecord['label']], 
                 marker     = markerOfMode[mode], 
                 linestyle  = 'None', 
                 markersize = 2, 
                 label      = resRecord['label'])  # Plot the conf' interval line
        curLegend = ax.legend (handles=[curLine], bbox_to_anchor=(-0.17, i*(1.1/len(resRecords)), 0., .102), loc='lower left', frameon=False)
        ax.add_artist (curLegend)
    
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(True)
    if zoomXlim!=None:
        plt.xlim (0, zoomXlim)
    else:
        plt.xlim (0, 2**cntrSize-1)        
    seaborn.despine(left=True, bottom=False, right=True)
    plt.show()

stdev           = 1
cntrSize8modes  = ['FP_e6', 'F2P_lr_h2', 'F2P_lr_h1', 'F2P_sr_h2', 'F2P_sr_h1', 'FP_e2', 'int']
cntrSize16modes = ['FP_e5', 'FP_e8', 'F2P_sr_h1', 'F2P_sr_h2', 'F2P_lr_h1', 'F2P_lr_h2', 'F2P_li_h1', 'F2P_li_h2'],  
cntrSize = 16
if cntrSize==16:
    modes = ['FP_e5', 'FP_e8', 'F2P_sr_h1', 'F2P_sr_h2', 'F2P_lr_h1', 'F2P_lr_h2', 'F2P_li_h1', 'F2P_li_h2', 'int']  
else:
    modes  = ['FP_e6', 'F2P_lr_h2', 'F2P_lr_h1', 'F2P_sr_h2', 'F2P_sr_h1', 'FP_e2', 'int', 'F2P_li_h1', 'F2P_li_h2']

verbose = [settings.VERBOSE_PCL]
if settings.VERBOSE_PCL in verbose:
    pclOutputFileName = ResFileParser.genMsePclFileName (cntrSize)
    if os.path.exists(f'../res/pcl_files/{pclOutputFileName}'):
        os.remove(f'../res/pcl_files/{pclOutputFileName}')
for df in [1, 10, 100, 1000]:
    simQuantErr (cntrSize       = cntrSize, 
                 modes          = modes, 
                 numPts         = 1000, 
                 stdev          = stdev,
                 dist           = 'Student',  
                 df             = df, #float('inf'),  # the df (degree of freedom) parameter; relevant only when using the t-student dist'.  
                 vecLowerBnd    = -4*stdev, 
                 vecUpperBnd    =  4*stdev,
                 # outLier        = 100*stdev,
                 verbose = verbose) #[settings.VERBOSE_RES, settings.VERBOSE_PLOT])  
# plotScaledGrids (zoomXlim=1, cntrSize=7, modes=['FP_e6', 'F2P_lr_h2', 'F2P_lr_h1', 'F2P_sr_h2', 'F2P_sr_h1', 'FP_e2', 'int'])

# scaled 'F2P_lr_h1' is identical to int.