import matplotlib, seaborn, pickle, os
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.pylab as pylab
import os, numpy as np, pandas as pd
from pandas._libs.tslibs import period
from printf import printf, printfDict
from nltk.corpus.reader import lin

import settings, SingleCntrSimulator
from settings import * 

# Color-blind friendly pallette
BLACK       = '#000000' 
ORANGE      = '#E69F00'
SKY_BLUE    = '#56B4E9'
GREEN       = '#009E73'
YELLOW      = '#F0E442'
BLUE        = '#0072B2'
VERMILION   = '#D55E00'
PURPLE      = '#CC79A7'

MARKER_SIZE = 16
MARKER_SIZE_SMALL = 1
LINE_WIDTH = 3 
LINE_WIDTH_SMALL = 2 
FONT_SIZE = 10
FONT_SIZE_SMALL = 5
LEGEND_FONT_SIZE = 10
LEGEND_FONT_SIZE_SMALL = 5 
USE_FRAME              = True # When True, plot a "frame" (box) around the plot 

# The markers used for each alg', in the dist' case
def markerOfMode (
        modeStr : str
    )   -> str:
    """
    Given a string defining a mode, return the color used when plotting the results of this mode.
    """
    return 'o'
    # {'F2P_li'    : 'o',
    #      'F2P_lr'    : 'o',
    #      'F2P_sr'    : 'o',
    #      'F3P'       : 'v',
    #      'SEAD stat' : '^',
    #      'SEAD_stat' : '^',
    #      'SEAD dyn'  : 's',
    #      'FP'        : 'p',
    #      'Tetra stat': 'p',
    #      'Tetra dyn' : 'X',
    #      'CEDAR'     : '<',
    #      'Morris'    : '>',
    #      'AEE'       : 'o'
    # }


def colorOfMode (
        modeStr : str
    ) -> str:
    """
    Given a string defining a mode, return the color used when plotting the results of this mode.
    """
    if modeStr.startswith('F3P_li_h2'):
        return PURPLE
    elif modeStr.startswith('F3P'):
        return BLACK
    elif modeStr=='SEAD_stat_e4':
        return 'black'
    elif modeStr.startswith('SEAD'):
        return VERMILION
    elif modeStr.startswith('FP'):
        return BLUE
    elif modeStr.startswith('Tetra'):
        return BLUE
    elif modeStr.startswith('CEDAR'):
        return BLUE
    elif modeStr.startswith('Morris'):
        return ORANGE
    elif modeStr.startswith('AEE'):
        return YELLOW
    elif modeStr.startswith('int'):
        return black
    elif modeStr.startswith('F2P_l'):
        return GREEN
    elif modeStr.startswith('F2P_s'):
        return PURPLE
    elif modeStr.startswith('F3P_l'):
        return BLACK
    elif modeStr.startswith('F3P_s'):
        return BLUE
    
colors = ['green', 'purple', 'brown', 'black', 'blue', 'yellow', 'magenta', 'red', 'green', 'purple', 'brown', 'black']

def labelOfDist (dist : str) -> str:
    """
    Given a distribution, return a label-string for it, to appear in the paper's plots. 
    """
    if dist=='uniform':
        return 'Uniform'
    elif dist=='norm':
        return 'Normal'
    elif dist.startswith('t_'):
        nu = int(dist.split('_')[1])
        return f't, $\\nu={nu}$'
    elif dist=='MobileNet_V2':
        return 'MNet\_V2'
    elif dist=='MobileNet_V3':
        return 'MNet\_V3'
    else:
        return dist

def genRndErrFileName (cntrSize : int) -> str:
    """
    Given the counter's size, generate the .pcl filename.
    """
    return f'rndErr_n{cntrSize}'

def labelOfMode (
        mode     : str,
        cntrSize : int = 8,
        ):
    """
    Generates a label string that details the counter's settings (param vals), to be used in plots.
    """
    if mode.startswith('F2P') or mode.startswith('F3P'):
        return genFxpLabel(mode)
    elif mode.startswith('FP'):
        expSize  = int(mode.split('_e')[1])
        mantSize = cntrSize - expSize - 1 
        return genFpLabel (expSize=expSize, mantSize=mantSize)
    elif mode=='int':
        return f'INT{cntrSize}'
    else:
        return mode

def genFpLabel (mantSize : int, expSize : int) -> str:
    """
    Generates a label string that details the counter's settings (param vals), to be used in plots.
    """
    return f'FP {mantSize}M{expSize}E'


def genFxpLabel (mode : str): # a mode describing the mode flavors
    """
    Given a string that details the parameters of F2P or F3P, generate a label string to be used in plots.
    """
    if mode.endswith('_ds'):
        mode = mode.split('_ds')[0]
    labelOfMode = {
    'F2P_lr_h2'     : r'F2P$_{LR}^2$',
    'F2P_sr_h2'     : r'F2P$_{SR}^2$',
    'F2P_li_h2'     : r'F2P$_{LI}^2$',
    'F2P_si_h2'     : r'F2P$_{SI}^2$',
    'F3P_lr_h2'     : r'F3P$_{LR}^2$',
    'F3P_sr_h2'     : r'F3P$_{SR}^2$',
    'F3P_li_h2'     : r'F3P$_{LI}^2$',
    'F3P_li_h3'     : r'F3P$_{LI}^3$',
    'F3P_si_h2'     : r'F3P$_{SI}^2$',
    'F2P_li_h2_ds'  : r'F2P$_{LI}^2$',
    }
    return labelOfMode[mode]
    # Trying to automate the code above:
    # flavor      = str(mode.split('_')[1]).upper()
    # hyperSize   = int(mode.split ('_h')[1])
    # print (f'flavor={flavor}, hyperSize={hyperSize}')
    # if flavor=='SR':
    #     if hyperSize==1:
    #         return r'F2P$_{SR}^1$'
    #         ...

def fxpSettingsToLabel (mode : str) -> str:
    """
    Given a string detailing the settings of F2P or F3P, return the corresponding label
    """
    numSettings = getFxpSettings(mode)
    return genFxpLabel (
        nSystem     = numSettings['nSystem'], 
        flavor      = numSettings['flavor'], 
        hyperSize   = numSettings['hyperSize']
    )

class ResFileParser (object):
    """
    Parse result files, and generate plots from them.
    """

    # Set the parameters of the plot (sizes of fonts, legend, ticks etc.).
    # mfc='none' makes the markers empty.
    setPltParams = lambda self, size = 'large': matplotlib.rcParams.update({
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
        'axes.titlesize'    :FONT_SIZE_SMALL
        })
    
    def __init__ (self):
        """
        Initialize a Res_file_parser, used to parse result files, and generate plots. 
        """
        # List of algorithms' names, used in the plots' legend, for the dist' case
        self.markers = ['o', 'v', '^', 's', 'p', 'X']
        self.points = []
        
    def rdPcl (
            self, 
            pclFileName : str,      
            exitError   : bool = False # When true, exit with error if the given pclFileName doesn't exist.
            ):
        """
        Given a .pcl filename, read all the data it contains into self.points
        """
        if not(settings.checkIfInputFileExists ('../res/pcl_files/{}' .format(pclFileName), exitError=exitError)):
            return
        pclFile = open('../res/pcl_files/{}' .format(pclFileName), 'rb')
        while 1:
            try:
                self.points.append(pickle.load(pclFile))
            except EOFError:
                break
            
    def dumpToPcl (self, pclOutputFileName):
        """
        Dump all the points in self.points to a .pcl. file
        """
    
        pclOutputFile = open(f'../res/pcl_files/{pclOutputFileName}', 'ab+')
    
        for point in self.points:
            pickle.dump(point, pclOutputFile) 

    def rmvFromPcl (
            self,
            pclFileName : str  = None,
            listOfDicts : list = [],
            verbose     : list = []            
        ):
        """
        Remove entries from a given .pcl. file
        """   
        self.rdPcl (pclFileName)

        pclFileNameWoExtension = pclFileName.split('.pcl')[0]
        if VERBOSE_RES in verbose:
            resFile = open (f'../res/{pclFileNameWoExtension}.res', 'w')
            for point in self.points:
                printf (resFile, f'{point}\n')

        for dict in listOfDicts:
            for key, value in dict.items():
                self.points = [point for point in self.points if point[key]!=value]
        # os.remove(f'../res/pcl_files/{pclFileName}')
        
        pclOutputFile = open (f'../res/pcl_files/{pclFileNameWoExtension}_.pcl', 'wb+')
        for point in self.points:
            pickle.dump(point, pclOutputFile) 
        if VERBOSE_RES in verbose:
            resFile = open (f'../res/{pclFileNameWoExtension}_.res', 'w')
            for point in self.points:
                printf (resFile, f'{point}\n')

    def printAllPoints (self, cntrSize=None, cntrMaxVal=None, printToScreen=False):
        """
        Format-print data found in self.points.
        Typically, self.points are filled earlier by data read from a .pcl or .res file.
        The points are printed into a '.dat' file, located in '/res' directory.
        if the input argument printToScreen==True, points are also printed to the screen.
        """
        if (cntrSize == None and cntrMaxVal != None) or (cntrSize != None and cntrMaxVal == None):
            settings.error ('ResFileParser.printAllPoints() should be called with either cntrSize and cntrMaxVal having both default value, or both having non-default- values.')
        if cntrSize == None and cntrMaxVal == None:
            outputFileName = '1cntr.dat' 
            datOutputFile = open ('../res/{}' .format (outputFileName), 'w')        
            points = [point for point in self.points]
        else:
            outputFileName = '1cntr_n{}_MaxVal{}.dat' .format (cntrSize, cntrMaxVal) 
            datOutputFile = open ('../res/{}' .format (outputFileName), 'w')        
            points = [point for point in self.points if (point['cntrSize'] == cntrSize and point['cntrMaxVal'] == cntrMaxVal)]
        for mode in [point['mode'] for point in points]:
            pointsOfThisMode = [point for point in points if point['mode'] == mode]
            for point in pointsOfThisMode:
                printf (datOutputFile, f'{point}\n\n')
            if printToScreen:
                for point in pointsOfThisMode:
                    print (point)

    def genErVsCntrSizePlot (
            self,
            erType         = 'Rd',
            numOfExps      = 50,
            modes          = ['F2P_li', 'CEDAR', 'Morris'], # 'SEAD_dyn'],
            minCntrSize    = 8,
            maxCntrSize    = 64,
        ):
        """
        Generate a plot showing the error as a function of the counter's size.
        This function is used show the results of single-cntr sim.
        """

        outputFileName = f'1cntr_PC_{erType}' 
        self.setPltParams ()  # set the plot's parameters (formats of lines, markers, legends etc.).
        _, ax = plt.subplots()
        points = [point for point in self.points if point['numOfExps'] == numOfExps and point['erType'].startswith(erType)]
        if len(points)==0: # no points to plot
            return

        for mode in modes:
            pointsOfThisMode = [point for point in points if point['mode'] == mode]
            if pointsOfThisMode == []:
                print (f'No points found for mode {mode} and numOfExps={numOfExps}')
                continue
            cntrSizes = [point['cntrSize'] for point in pointsOfThisMode if (point['cntrSize'] >= minCntrSize and point['cntrSize'] <= maxCntrSize)]
            y = []
            for cntrSize in cntrSizes:
                pointOfThisModeNCntrSize = [point for point in pointsOfThisMode if point['cntrSize'] == cntrSize]
                if len(pointOfThisModeNCntrSize) == 0:
                    settings.error (f'No points for mode={mode}, cntrSize={cntrSize}')
                elif len(pointOfThisModeNCntrSize) > 1:
                    print (f'Note: found more than one point for mode={mode}, cntrSize={cntrSize}. The points are') 
                    print (pointOfThisModeNCntrSize[0]) 
                    print (pointOfThisModeNCntrSize[1]) 
                point = pointOfThisModeNCntrSize[0]
                y_lo, y_avg, y_hi = point['Lo'], point['Avg'], point['Hi']                     
                ax.plot ((cntrSize, cntrSize), (y_lo, y_hi), color=colorOfMode(mode))  # Plot the conf' interval line
                y.append (y_avg)
            ax.plot (cntrSizes, y, color=colorOfMode(mode), marker=markerOfMode(mode),
                     markersize=MARKER_SIZE_SMALL, linewidth=LINE_WIDTH, label=point['mode'], mfc='none') 
        
        plt.xlabel('Counter Size [bits]')
        plt.ylabel('RMSE')
        plt.yscale ('log')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend (by_label.values(), by_label.keys(), fontsize=LEGEND_FONT_SIZE, frameon=False)        
        if not(USE_FRAME):
            seaborn.despine(left=True, bottom=True, right=True)
        plt.savefig ('../res/{}.pdf' .format (outputFileName), bbox_inches='tight')        
    
    def genErVsCntrSizeTable (
            self,
            datOutputFile,
            numOfExps   : int  = 50,
            modes       : list = ['CEDAR', 'Morris', 'SEAD_dyn'],
            cntrSizes   : list = [],
            statType    : str  = 'Mse',
            erType      : str  = 'Rd',
            rel_abs_n   : bool = False, # When True, consider relative errors, Else, consider absolute errors.
            width       : int  = None,  # The width of the CMS. Relevant only for CMS' sim results.
            maxValBy    : str  = None,
            normalizeByPerfectCntr : bool = True, # when True, normalize all mode's results by dividing them by the value obtained by a perfect counter
            normalizeByMinimal     : bool = True, # when True, normalize all mode's results by dividing them by the value obtained by the lowest value at this row
        ):
        """
        Generate a table showing the error as a function of the counter's size.
        This function is used show the results of a single-cntr sim.
        """
    
        points = [point for point in self.points if point['numOfExps'] == numOfExps and point['statType']==statType and point['rel_abs_n']==rel_abs_n]
        if width!=None:
            points = [point for point in points if point['width']==width]
        
        if len(points)==0:
            warning (f'No points found for numOfExps={numOfExps}, cntrSize={cntrSize}, statType={statType}, rel_abs_not={rel_abs_n}, width={width}')
            return     
        printf (datOutputFile, '\t')
        if maxValBy!=None:
            printf (datOutputFile, f'cntrMaxVal\t')
        for mode in modes:
            printf (datOutputFile, f'{mode}\t')
            if mode!=modes[-1]:
                printf (datOutputFile, '& ')
        printf (datOutputFile, ' \\\\ \n')
        for cntrSize in cntrSizes:
            pointsOfThisCntrSize = [point for point in points if point['cntrSize']==cntrSize]
            printf (datOutputFile, f'{cntrSize} & ')
            if maxValBy!=None:
                cntrMaxVal = int (SingleCntrSimulator.getCntrMaxValFromFxpStr (
                    cntrSize = cntrSize,
                    fxpSettingStr = maxValBy 
                ))
                printf (datOutputFile, f'{cntrMaxVal} &\t')
                pointsOfThisCntrSize = [point for point in pointsOfThisCntrSize if point['cntrMaxVal']==cntrMaxVal]
            pointsOfThisCntrSizeErType = [point for point in pointsOfThisCntrSize if point['erType'].startswith(erType)]  
            if pointsOfThisCntrSizeErType == []:
                warning (f'No points found for numOfExps={numOfExps}, cntrSize={cntrSize}, , statType={statType}, rel_abs_not={rel_abs_n}, erType={erType}, width={width}')
                continue
            minVal = min ([point['Avg'] for point in pointsOfThisCntrSizeErType if point['mode']!='PerfectCounter'])
            if normalizeByPerfectCntr:
                pointsOfPerfCntr = [point for point in pointsOfThisCntrSize if point['mode']=='PerfectCounter']
                if len(pointsOfPerfCntr)==0:
                    error (f'In ResFileParser.genErVsCntrSizeTable(). Requested normalizeByPerfectCntr, but no such points for cntrSize={cntrSize}')
                elif len(pointsOfPerfCntr)>1:
                    warning (f'In ResFileParser.genErVsCntrSizeTable(). Multiple points for cntrSize={cntrSize}')
                valOfPerfCntr = pointsOfPerfCntr[0]['Avg']
                minVal = min(pointsOfThisCntrSizeErType)
            for mode in modes:
                pointsToPrint = [point for point in pointsOfThisCntrSizeErType if point['mode'] == mode]
                if pointsToPrint == []:
                    warning (f'In ResFilePerser.genErVsCntrSizeTable(). No points found for numOfExps={numOfExps}, cntrSize={cntrSize}, mode={mode}, width={width}, statType={statType}, rel_abs_n={rel_abs_n}')
                    if mode!=modes[-1]:
                        printf (datOutputFile, ' & ')
                    continue
                if len(pointsToPrint)>1:
                    warning (f'found {len(pointsToPrint)} points for numOfExps={numOfExps}, cntrSize={cntrSize}, mode={mode}, width={width}, statType={statType}, rel_abs_n={rel_abs_n}')
                val2print = pointsToPrint[0]['Avg']
                if normalizeByPerfectCntr:
                    val2print /= valOfPerfCntr
                if normalizeByMinimal:
                    val2print /= minVal
                if pointsToPrint[0]['Avg']<=minVal*1.01:
                    printf (datOutputFile, '\\green{\\textbf{')
                    if normalizeByMinimal:
                        printf (datOutputFile, '{:.2f}' .format(val2print))
                    else:
                        printf (datOutputFile, '{:.2e}' .format(val2print))
                    printf (datOutputFile, '}}')
                else:
                    if normalizeByMinimal:
                        printf (datOutputFile, '{:.2f}' .format(val2print))
                    else:
                        printf (datOutputFile, '{:.2e}' .format(val2print))
                if mode!=modes[-1]:
                    printf (datOutputFile, ' & ')
            printf (datOutputFile, ' \\\\\n')
    
    def rmvDuplicatedPoints (
            self,
            pclFileName : str  = None,
            verbose     : list = []            
        ):
        """
        Remove duplicated points from the results. 
        Save the results while leaving only a single example of each setting.
        """
        self.rdPcl (pclFileName)

        pclFileNameWoExtension = pclFileName.split('.pcl')[0]

        i = 0
        while i < len(self.points):
            point = self.points[i]
            j = i + 1
            while j < len(self.points):
                suspectedPoint = self.points[j]
                if point['statType']  == suspectedPoint['statType']  and \
                   point['confLvl']   == suspectedPoint['confLvl']   and  \
                   point['numOfExps'] == suspectedPoint['numOfExps'] and \
                   point['numIncs']   == suspectedPoint['numIncs']   and \
                   point['mode']      == suspectedPoint['mode']      and \
                   point['cntrSize']  == suspectedPoint['cntrSize']  and \
                   point['depth']     == suspectedPoint['depth']     and \
                   point['width']     == suspectedPoint['width']     and \
                   point['numFlows']  == suspectedPoint['numFlows']  and \
                   point['seed']      == suspectedPoint['seed']      and \
                   point['maxValBy']  == suspectedPoint['maxValBy']  and \
                   point['dwnSmpl']   == suspectedPoint['dwnSmpl']   and \
                   point['rel_abs_n'] == suspectedPoint['rel_abs_n']:
                   self.points.pop(j)
                else:
                   j += 1
            i += 1
 
        pclOutputFile = open (f'../res/pcl_files/{pclFileNameWoExtension}_.pcl', 'wb+')
        for point in self.points:
            pickle.dump(point, pclOutputFile) 
        
        # pclOutputFile = open (f'../res/pcl_files/{pclFileNameWoExtension}_.pcl', 'wb+')
        # for point in self.points:
        #     pickle.dump(point, pclOutputFile) 
        # if VERBOSE_RES in verbose:
        #     resFile = open (f'../res/{pclFileNameWoExtension}_.res', 'w')
        #     for point in self.points:
        #         printf (resFile, f'{point}\n')
        
    
    
    def genErVsMemSizePlotCms (
            self,
            traceName   : str  = 'Caida1',
            numOfExps   : int  = 10,
            cntrSize    : int  = 8,
            depth       : int  = 4,
            statType    : str  = 'normRmse',
            rel_abs_n   : bool = False, # When True, consider relative errors, Else, consider absolute errors.
            maxValByStr : str  = 'F2P_li_h2', # The mode by which the counter's max value is determined. 
            numIncs     : int  = 100000000, # number of increments
            ignoreModes : list = [],# List of modes to NOT include in the plot
        ):
        """
        Generate a plot showing the error as a function of the overall memory size.
        This function is used to show the results of CMS (Count Min Sketch) simulations.        
        """
    
        dupsFileName = 'listOfDuplicateEntries.txt'
        dupsFile = open (f'../res/{dupsFileName}', 'w')
        foundDups = False
        self.setPltParams ()  # set the plot's parameters (formats of lines, markers, legends etc.).
        _, ax = plt.subplots()
        points = [point for point in self.points if 
                  point['numOfExps'] == numOfExps and 
                  point['statType']  == statType  and 
                  point['rel_abs_n'] == rel_abs_n and 
                  point['cntrSize']  == cntrSize  and
                  point['numIncs']   == numIncs   
                  ]
        if points == []:
            warning (f'In ResFileParser.genErVsMemSizePlotCms(). No points found for numOfExps={numOfExps}, cntrSize={cntrSize}, maxValBy={maxValByStr}, numIncs={numIncs}, statType={statType}, rel_abs_n={rel_abs_n}')
            return
        modes = [point['mode'] for point in points]
        modes = list(set([mode for mode in modes if mode not in ignoreModes])) # uniquify and filter-out undesired modes 
        minY, maxY = float('inf'), 0 #lowest, highest Y-axis values; to be used for defining the plot's scaling
        
        for mode in modes:
            pointsOfMode = [point for point in points if point['mode'] == mode]
            widths = [point['width'] for point in pointsOfMode]
            minMemSizeInKB  = 10**0
            minMemSize      = minMemSizeInKB * KB  
            widths = sorted(list(set([w for w in widths if w>=(minMemSize/4)])))
            y    = []
            xVec = []
            for width in widths:
                pointsToPlot = [point for point in pointsOfMode if point['width']==width]
                if len(pointsToPlot)>1:
                    warning (f'found {len(pointsToPlot)} points for numOfExps={numOfExps}, cntrSize={cntrSize}, mode={mode}, width={width}, statType={statType}, rel_abs_n={rel_abs_n}. Printing the duplicated points to ../res/debug.txt')
                    foundDups = True
                    for point in pointsToPlot: 
                        printfDict (dupsFile, point)
                point = pointsToPlot[0]
                y.append(point['Avg'])
                x = depth*width*(cntrSize/8)/KB # memory consumption, in KB
                ax.plot ( # Plot the confidence-interval of this point
                    (x, x), 
                    (point['Lo'], point['Hi']),
                    color       = colorOfMode(mode),
                    # marker      = '+', 
                    # markersize  = MARKER_SIZE, 
                    # linewidth   = LINE_WIDTH, 
                    # label       = labelOfMode(point['mode']), 
                    # mfc         ='none',
                    # linestyle   ='None'
                )
                xVec.append (x) 
                minY = min (minY, point['Avg']) # lowest Y-axis value; to be used for defining the plot's scaling
                maxY = max (maxY, point['Avg']) # highest Y-axis value; to be used for defining the plot's scaling
            memSizesInKb = [depth*w*(cntrSize/8)/KB for w in widths] # Memory size in KB = width * depth / 1024.
            ax.plot ( # Plot the averages values
                xVec,
                y,
                color       = colorOfMode(mode),
                linewidth   = LINE_WIDTH, 
                label       = labelOfMode(point['mode']), 
                mfc         ='none',
                # marker      = '+', 
                # markersize  = MARKER_SIZE, 
                # linestyle   ='None'
            ) 

        plt.xlabel('Memory [KB]')
        if statType=='normRmse':
            plt.ylabel('Normalized RMSE')
        else:
            plt.ylabel(statType)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend (by_label.values(), by_label.keys(), fontsize=LEGEND_FONT_SIZE, frameon=False)
        plt.xlim ([0.95*min(memSizesInKb), 10**3])       
        plt.xscale ('log')
        outputFileName = 'cms_{}_n{}_{}_by_{}' .format (traceName, cntrSize, 'rel' if rel_abs_n else 'abs', maxValByStr) 
        if not(USE_FRAME):
            seaborn.despine(left=True, bottom=True, right=True)
        plt.savefig (f'../res/{outputFileName}.pdf', bbox_inches='tight')        
        dupsFile.close()
        if not(foundDups) and os.path.isfile (f'../res/{dupsFileName}'):
            os.remove(f'../res/{dupsFileName}') 
                
    def genErVsMemSizePlotSpaceSaving (
            self,
            traceName   : str  = 'Caida1',
            numOfExps   : int  = 10,
            cntrSize    : int  = 8,
            depth       : int  = 4,
            statType    : str  = 'normRmse',
            rel_abs_n   : bool = False, # When True, consider relative errors, Else, consider absolute errors.
            ignoreModes : list = [],# List of modes to NOT include in the plot
        ):
        """
        Generate a plot showing the error as a function of the counter's size.
        This function is used to show the results of Space-Saving simulations.        
        """
    
        dupsFileName = 'listOfDuplicateEntries.txt'
        dupsFile = open (f'../res/{dupsFileName}', 'w')
        foundDups = False
        self.setPltParams ()  # set the plot's parameters (formats of lines, markers, legends etc.).
        _, ax = plt.subplots()

        points = [point for point in self.points if 
                  point['numOfExps'] == numOfExps and 
                  point['statType']  == statType  and 
                  point['rel_abs_n'] == rel_abs_n]
        if points == []:
            warning (f'No points found for numOfExps={numOfExps}, cntrSize={cntrSize}')
            return
        modes = [point['mode'] for point in points]
        modes = list(set([mode for mode in modes if mode not in ignoreModes]))
        minY, maxY = float('inf'), 0 
        for mode in modes:
            pointsOfMode = [point for point in points if point['mode'] == mode]
            memSizes = [point['cacheSize'] for point in pointsOfMode]
            minMemSizeInKB  = 10**0
            minMemSize      = minMemSizeInKB * KB  
            memSizes        = [m for m in memSizes if m>=minMemSize]
            if len(memSizes)==0:
                warning (f'No points found for numOfExps={numOfExps}, cntrSize={cntrSize}, mode={mode}')
            y = []
            xVec = []
            for memSize in memSizes:
                pointsToPlot = [point for point in pointsOfMode if point['cacheSize']==memSize]
                if len(pointsToPlot)>1:
                    warning (f'found {len(pointsToPlot)} points for numOfExps={numOfExps}, mode={mode}, memSize={memSize}, statType={statType}, rel_abs_n={rel_abs_n}. Printing the duplicated points to ../res/debug.txt')
                    foundDups = True
                    for point in pointsToPlot: 
                        printfDict (dupsFile, point)
                point = pointsToPlot[0]
                y.append(point['Avg'])
                xVec.append (x) 
                ax.plot (
                    (memSize, memSize), 
                    (point['Lo'], point['Hi']),
                    color       = colorOfMode(mode),
                ) 
                minY = min (minY, point['Avg'])
                maxY = max (maxY, point['Avg'])
            memSizesInKb = [memSize/KB for memSize in memSizes] # Memory size in KB = width * depth / 1024. 
            ax.plot (
                xVec, 
                y, 
                color       = colorOfMode(mode),
                linewidth   = LINE_WIDTH, 
                label       = labelOfMode(point['mode']), 
                mfc         ='none',
            ) 
        
        plt.xlabel('Memory [KB]')
        plt.ylabel('Normalized RMSE')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend (by_label.values(), by_label.keys(), fontsize=LEGEND_FONT_SIZE, frameon=False)
        plt.xlim ([0.95*min(memSizesInKb), 10**3])       
        plt.xscale ('log')          
        outputFileName = 'ss_{}_n{}_{}' .format (traceName, cntrSize, 'rel' if rel_abs_n else 'abs') 
        if not(USE_FRAME):
            seaborn.despine(left=True, bottom=True, right=True)
        plt.savefig (f'../res/{outputFileName}.pdf', bbox_inches='tight')        
        dupsFile.close()
        if not(foundDups) and os.path.isfile (f'../res/{dupsFileName}'):
            os.remove(f'../res/{dupsFileName}')            
                
    def genErVsCntrMaxValPlot (
            self, 
            cntrSize    = 8, # size of the compared counters.
            plotAbsEr   = True # if True, plot the absolute errors. Else, plot the relative errors
        ):
        """
        Generate a plot showing the relative / abs err as a function of the maximum counted value
        """

        outputFileName = '1cntr_{}_n{}' .format ('abs' if plotAbsEr else 'rel', cntrSize) 
        datOutputFile = open ('../res/{}.dat' .format (outputFileName), 'w')        
        printf (datOutputFile, 'cntrSize={}\n' .format (cntrSize))                    

        self.setPltParams ()  # set the plot's parameters (formats of lines, markers, legends etc.).
        _, ax = plt.subplots()
        preferredModes = ['F2P_li', 'Tetra stat', 'SEAD stat', 'SEAD dyn', 'CEDAR', 'Morris']
        for mode in [point['mode'] for point in self.points if point['mode'] in preferredModes]:
            pointsOfThisMode = [point for point in self.points if (point['mode'] == mode and point['cntrSize'] == cntrSize)]
            cntrMaxVals = sorted ([point['cntrMaxVal'] for point in pointsOfThisMode])
            y = []
            for cntrMaxVal in [item for item in cntrMaxVals if (item >= 10000)]:
                pointOfThisModeNMaxVal = [point for point in pointsOfThisMode if point['cntrMaxVal'] == cntrMaxVal]
                pointOfThisModeNMaxVal = [point for point in pointOfThisModeNMaxVal if point['settingStr'] not in ['F2P_n8_h1', 'SEADstat_n8_e1']]  # $$$
                if (len(pointOfThisModeNMaxVal) != 1):
                    print ('bug at genErVsCntrMaxValPlot: pointOfThisModeNMaxVal!=1. Points are')
                    print (pointOfThisModeNMaxVal) 
                    exit ()
                point = pointOfThisModeNMaxVal[0]
                
                if (plotAbsEr):
                    y_lo, y_avg, y_hi = point['absRdErLo'], point['absRdErAvg'], point['absRdErHi']
                else:
                    y_lo, y_avg, y_hi = point['relRdErLo'], point['relRdErAvg'], point['relRdErHi']
                     
                printf (datOutputFile, 'settingStr={}, mode={}. cntrMaxVal={}, y_lo={:.2f}, y_hi={:.2f}, y_avg={:.2f}\n' .format 
                                       (point['settingStr'], mode, cntrMaxVal, y_lo, y_hi, y_avg))                    
                y.append (y_avg)

            ax.plot (cntrMaxVals, y, color=colorOfMode(mode), marker=markerOfMode(mode),
                     markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=point['settingStr'], mfc='none') 

        plt.xlabel('Counter Maximum Value')
        plt.ylabel('Avg. {} Eror' .format ('Absolute' if plotAbsEr else 'Relative'))
        ax.set_xscale ('log')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend (by_label.values(), by_label.keys(), fontsize=LEGEND_FONT_SIZE) #, frameon=False)        
        plt.savefig ('../res/{}.pdf' .format (outputFileName), bbox_inches='tight')        
        
       
    def genResolutionPlotByModes (self,
            cntrSize        = 8,    # cntrSizes for which the plot will be generated. 
            minCntrVal      = 0,  # min' X (counter) value at the plot
            xLog            = False,    # When True, plot the x axis in a log' scaling.
            ignoreModes     = [],       # modes to ignore (do NOT plot for these modes)
            ) -> None:
        """
        Generate a plot showing the resolution as a function of the counted val for the given modes
        """
        self.setPltParams   ()  # set the plot's parameters (formats of lines, markers, legends etc.).
        _, ax = plt.subplots()
        points = [point for point in self.points if point['cntrSize'] == cntrSize]
        modes = list(set([point['mode'] for point in points]))
        modes = [item for item in modes if not(item in ignoreModes)]
        xMaxVals = np.zeros (len(modes))
        yMinVal  = float('inf')
        yMaxVal  = 0
        for i in range(len(modes)):
            mode = modes[i]
            pointsOfThisMode = [point for point in points if point['mode'] == mode]
            if pointsOfThisMode == []:
                print (f'No points found for mode {mode}')
                continue
            if len(pointsOfThisMode) > 1:
                settings.error (f'More than a single list of points for mode {mode}')
            points2plot = pointsOfThisMode[0]['points']
            xMaxVals[i] = points2plot['X'][-1] # the array should be sorted; the last val is the largest val
            yMaxVal = max (yMaxVal, np.max(points2plot['Y']))
            yMinVal = min (yMinVal, np.min(points2plot['Y']))
            ax.plot (
                points2plot['X'], 
                points2plot['Y'], 
                color       = colorOfMode(mode), 
                marker      = markerOfMode(mode),
                markersize  = MARKER_SIZE_SMALL, 
                linewidth   = LINE_WIDTH_SMALL,
                # linestyle   = '', 
                label       = labelOfMode(mode), 
                mfc         = 'none'
            ) 

        plt.xlabel('Counted Value')
        plt.ylabel(f'Relative Resolution')
        plt.yscale ('log')
        if xLog: 
            plt.xscale ('log')
            plt.xlim ([1, 0.99*np.min(xMaxVals)])
        else:
            plt.xlim ([minCntrVal, np.min(xMaxVals)])
        plt.ylim (0.99*yMinVal, 1.01*yMaxVal) 
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend (by_label.values(), by_label.keys(), fontsize=LEGEND_FONT_SIZE, frameon=False)        
        plt.savefig ('../res/ResolutionByModes_n{}_{}.pdf' .format (cntrSize, 'log' if xLog else 'lin'), bbox_inches='tight')        


    def genResolutionPlotBySettingStrs (
            self,
            settingStrs    = [],           # Concrete settings for which the plot will be generated
            minCntrVal      = 1,            # min' X (counter) value at the plot
            maxCntrVal      = float('inf'), # max X (counter) value at the plot
            xLog            = False,        # When True, plot the x axis in a log' scaling.
        ) -> None:
        """
        Generate a plot showing the resolution as a function of the counted val for the given settings.
        Each input setting string details the cntrSize, exponent size (expSize), hyper-exp size (hyperSize), etc.
        """
        self.setPltParams   ()  # set the plot's parameters (formats of lines, markers, legends etc.).
        _, ax = plt.subplots()
        colorIdx = 0
        for settingStr in settingStrs:
            pointsOfThisSettingStr = [point for point in self.points if point['settingStr'] == settingStr]
            if pointsOfThisSettingStr == []:
                print (f'No points found for settingStr {settingStr}')
                continue
            if len(pointsOfThisSettingStr) < 1:
                settings.error (f'More than a single list of points for settingStr {pointsOfThisSettingStr}')
            points      = pointsOfThisSettingStr[0]['points']
            params      = settings.extractParamsFromSettingStr (settingStr)
            mode        = params['mode']
            cntrSize    = params['cntrSize']
            ax.plot (points['X'], points['Y'], color=colors[colorIdx], marker=self.markers[colorIdx],
                     markersize=MARKER_SIZE_SMALL, linewidth=LINE_WIDTH_SMALL, label=labelOfMode(mode), mfc='none')
            colorIdx += 1 

        plt.xlabel('Counted Value')
        plt.ylabel(f'Relative Resolution')
        plt.yscale ('log')
        if xLog: 
            plt.xscale ('log')

        conf        = settings.getConfByCntrSize (cntrSize=cntrSize)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend (by_label.values(), by_label.keys(), fontsize=LEGEND_FONT_SIZE, frameon=False)        
        plt.savefig ('../res/resolutionBySettingStrs_n{}_{}.pdf' .format (cntrSize, 'log' if xLog else 'lin'), bbox_inches='tight')        


    def genErrByDfGraph (self,
                     cntrSize   : int  = 8,
                     resTypeStr : str  = 'abs', # a string detailing the y value for which the func' will generate a plot
                     numPts     : int  = None,         # num of points in the experiment
                     verbose    : list =[]
                 ):
        """
        Generate and save a plot of the Mean Square Error as a function of the Student's distribution df (degree of freedom) parameter. 
        """
        points = [point for point in self.points if point['dist']=='t']
        if numPts!=None:
            points = [point for point in points if point['numPts']==numPts]
        
        if VERBOSE_RES in verbose:
            resFile = open (f'../res/student_rndErr.res', 'a+')
            printf (resFile, f'// cntrSize={cntrSize}, errType={resTypeStr}\n')
            printedDfs = False
        _, ax = plt.subplots()
        printedDFs = False
        modes = sorted(list(set([point['mode'] for point in points])))
        for mode in modes: 
            pointsOfThisMode  = [point for point in points if point['mode']==mode]
            pointsOfThisMode  = sorted (pointsOfThisMode, key = lambda item : item['df']) # sort the points by their df value
            dfsWithThisMode   = [point['df'] for point in pointsOfThisMode]
            yVals             = [point[f'{resTypeStr}'] for point in pointsOfThisMode]
            if VERBOSE_RES in verbose:
                if not printedDFs:
                    printf (resFile, '\t\t\t')
                    for df in dfsWithThisMode:
                        printf (resFile, f'{df}\t\t\t')
                    printf (resFile, '\n')
                    printedDFs = True
                printf (resFile, f'{mode}\t')
                if mode=='int':
                    printf (resFile, '\t\t')
                for yVal in yVals:
                    printf (resFile, '{:.2e}\t' .format (yVal))
                printf (resFile, '\n')
            if settings.VERBOSE_PLOT in verbose: 
                ax.plot (dfsWithThisLabel, yVals, label=label)

        if VERBOSE_RES in verbose:
            printf (resFile, '\n')
        if settings.VERBOSE_PLOT not in verbose:
            return
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend (by_label.values(), by_label.keys(), fontsize=10, frameon=False)
        plt.xlim (1, 100)
        plt.xlabel (f'df')
        plt.ylabel (f'{resTypeStr}')
        plt.yscale ('log')
        # plt.show ()
        plt.savefig (f'../res/student_rndErr_n{cntrSize}_{resTypeStr}.pdf', bbox_inches='tight')        

    def ErrByDistBar (
            self,
            stdev       : float = 1,
            dist        : str   = 't_5',
            resTypeStr  : str   = 'abs', # a string detailing the y value for which the func' will generate a plot
            numPts      : int   = None,         # num of points in the experiment
            cntrSize    : int   = 8,
            modes       : list  = [], # modes of distributions to consider for the plot/res file.
            verbose     : list  = []
            ):
        """
        Generate and save a bar-plot of the Mean Square Error for the requested error types and distributions.
        Based on the verbose, the function may either save a bar plot and/or write the data to a formatted .res file. 
        """
        points = [point for point in self.points if point['stdev']==stdev]
        if numPts!=None:
            points = [point for point in points if point['numPts']==numPts]
        
        if VERBOSE_RES in verbose:
            resFile = open (f'../res/{genRndErrFileName(cntrSize)}.res', 'a+')
            printf (resFile, f'// errType={resTypeStr}, dist={dist}\n')
            printedDfs = False

        _, ax = plt.subplots()
        printedDFs = False
        
        if dist.startswith('t'): # For a student distribution, need to pick the points with the desired df
            dist_split = dist.split('_')
            if len(dist_split)<1:
                settings.error ('In ResFileParser.ErrByDistBar(): Student dist name should be in the format Student_df')
            df   = int(dist_split[1])
            pointsOfThisDist = [point for point in points if point['dist']=='t' and point['df']==df]  
        else: # For distributions others than "Student", need to pick the points belonging to that dist'
            pointsOfThisDist = [point for point in points if point['dist']==dist]  
        yVals = np.empty (len(modes))
        for i in range(len(modes)):
            mode = modes[i]
            if resTypeStr=='rel' and mode.endswith ('_h1'): # for relative errors, 'h1' is always worse than 'h2', so we can skip them for the plot. 
                continue
            pointsOfThisMode = [point for point in pointsOfThisDist if point['mode']==mode]
            if len(pointsOfThisMode)<1:
                print (f'In ErrByDistBar(): no points for dist Student, df={df}, mode={mode}')
                continue
            yVals[i] = pointsOfThisMode[0][f'{resTypeStr}']
            if VERBOSE_RES in verbose:
                printf (resFile, f'{mode}\t\t')
                if mode=='int' or mode.startswith('FP'):
                    printf (resFile, '\t\t')
                printf (resFile, '\t{:.2e}\n' .format (yVals[i]))
            if mode.startswith ('FP'):
                expSize = int(mode.split ('_e')[1])
                label = genFpLabel(expSize=expSize, mantSize=cntrSize-1-expSize)
            else:
                label = mode
            plt.bar (label, yVals[i]) #width=1 


        if VERBOSE_RES in verbose:
            printf (resFile, '\n')
        if settings.VERBOSE_PLOT not in verbose:
            return
        
        plt.ylabel (f'|R(x)|')
        plt.xticks (rotation=90)
        if resTypeStr=='rel':
            plt.yscale ('log')
        plt.savefig (f'../res/{resTypeStr}_{genRndErrFileName(cntrSize)}_{dist}.pdf', bbox_inches='tight')
        
    
    def optModeOfDist (
            self,
            cntrSize    : int  = 8,
            errType     : str  = 'abs', # Currently, either 'abs' (absolute) or 'rel' (relative
            distStr     : str  = 'uniform', 
            ignoreModes : list = [], # list of modes to ignore when extracting the results 
            onlyF2P     : bool = False, # When True, consider only F2P flavors
            onlyNonF2P  : bool = False, # When True, consider only non-F2P flavors
            ) -> list:
        """
        Find in the .pcl files the mode (e.g., FP_2e, F2P_li_h2) that minimizes the error for the given distribution.
        """
        
        self.rdPcl (f'{genRndErrFileName(cntrSize)}.pcl')
        points = [point for point in self.points if point['dist']==distStr]
        if onlyF2P:
            points = [point for point in points if point['mode'].startswith('F2P')]
        elif onlyNonF2P:
            points = [point for point in points if not(point['mode'].startswith('F2P'))]
            points = [point for point in points if point['mode'] not in ignoreModes] 
        points = sorted (points, key = lambda item : item[f'{errType}'])
        if len(points)==0:
            print (f'In ResFileParser.optModeOfDist(). No points found for cntrSize={cntrSize}, errType={errType}, dist={distStr}')
            return None
        return ([points[0]['mode'], points[0][f'{errType}']])


    def printRndErrTableRow (
            self,
            resFile,
            distStrs : list = ['uniform', 'norm', 't_5', 't_8', 'Resnet18', 'Resnet50', 'MobileNet_V2', 'MobileNet_V3'],
            cntrSize : int  = 8,
            errType  : str  = 'absMse'
            ):
        """
        Print a row in the table of quantization's rounding errors.
        """
        
        self.rdPcl (f'{genRndErrFileName(cntrSize)}.pcl')
        modes = settings.modesOfCntrSize(cntrSize)
        for mode in modes:
            printf (resFile, f'{mode} \t\t&') 
        printf (resFile, '\n')

        points = [point for point in self.points if point['mode'] in modes]
        
        for dist in distStrs:
            pointsOfThisDist = [point for point in points if point['dist']==dist]
            if len(pointsOfThisDist)==0:
                settings.error (f'In ResFileParser.optModeOfDist(). No points found for cntrSize={cntrSize}, errType={errType}, dist={dist}')
            minErr = min ([point[errType] for point in pointsOfThisDist])
            printf (resFile, f'\t\t{labelOfDist(dist)} & ') 
            for mode in modes:
                pointsOfThisDistAndMode = [point for point in pointsOfThisDist if point['mode']==mode]
                if len(pointsOfThisDistAndMode)==0:
                    print (f'In ResFileParser.optModeOfDist(). No points found for cntrSize={cntrSize}, errType={errType}, dist={dist}, mode={mode}')
                    printf (resFile, 'None & ')
                    continue
                val = pointsOfThisDistAndMode[0][errType]/minErr
                if val<1.01:
                    printf (resFile, '\\green{\\textbf{')
                    printf (resFile, '{:.1f}' .format (val))
                    printf (resFile, '}}')
                # elif val<100:
                #     printf (resFile, '{:.1f}' .format (val))
                elif val<10000:
                    printf (resFile, '{:.1f}' .format (val))
                else:
                    printf (resFile, '{:.1e}' .format (val))
                if mode!=modes[-1]:
                    printf (resFile, ' & ' .format (pointsOfThisDistAndMode[0][errType]/minErr))
            printf (resFile, ' \\\\ \n')

def genResolutionPlot ():
    """
    Generate plots showing the resolution as a function of the counted val for the given modes
    """
    
    my_ResFileParser = ResFileParser ()
    byModes = True
    if byModes:
        my_ResFileParser.rdPcl (pclFileName=f'resolutionByModes_F3P.pcl')
        for cntrSize in [8]:  # , 12, 16]:
            my_ResFileParser.genResolutionPlotByModes (
                # ignoreModes = ['CEDAR'],
                minCntrVal  = 1000,
                cntrSize    = cntrSize,
                xLog        = True
                )
    else:
        my_ResFileParser.rdPcl (pclFileName=f'resolutionBySettingStrs.pcl')
        my_ResFileParser.genResolutionPlotBySettingStrs(
            # settingStrs = ['FP_n7_m2_e5', 'F2Plr_n7_h1'], 
            #['FP_n7_m2_e5', 'FP_n7_m5_e2', 'F2Plr_n7_h2', 'F2Psr_n7_h2'], 
            # settingStrs = ['FP_n15_m13_e2', 'F2Psr_n15_h2'], # ['FP_n7_m5_e2', 'F2Psr_n7_h1'], 
            settingStrs = ['FP_n15_m10_e5', 'F2Plr_n15_h2'], # ['FP_n7_m5_e2', 'F2Psr_n7_h1'], 
            xLog        = True
            )

def genErrByDfGraph ():
    """
    Plot the Err as a func' of the df value at the Student-t dist'.
    """
    for cntrSize in [8, 16]:
        myResFileParser = ResFileParser ()
        pclFileName = genRndErrFileName (cntrSize) 
        myResFileParser.rdPcl (pclFileName)
        myResFileParser.genErrByDfGraph (cntrSize=cntrSize, resTypeStr='abs', verbose=[VERBOSE_RES])
        myResFileParser.genErrByDfGraph (cntrSize=cntrSize, resTypeStr='relMse', verbose=[VERBOSE_RES])

def genErrByDistBar ():
    """
    Generate and save a bar-plot of the Mean Square Error for the various distributions. 
    """
    for cntrSize in [8]: #, 16, 19]:
        myResFileParser = ResFileParser ()
        myResFileParser.rdPcl (f'{genRndErrFileName(cntrSize)}.pcl')
        for dist in ['uniform']: #, 'Gaussian', 't_5', 't_8']:
            for resTypeStr in ['rel']: #, 'abs']: # a string detailing the y value for which the func' will generate a plot
                myResFileParser.genErrByDistBar (
                    stdev       = 1,
                    dist        = dist,
                    cntrSize    = cntrSize,
                    modes       = settings.modesOfCntrSize (cntrSize),
                    resTypeStr  = resTypeStr,
                    numPts      = None,         # num of points in the experiment
                    verbose     =[VERBOSE_RES, settings.VERBOSE_PLOT]
                )
    
def genQuantErrTable ():
    """
    Print a formatted table detailing the quantization's rounding  errors.
    """
    resFile = open ('../res/errTable.dat', 'a+')
    for cntrSize in [8, 16, 19]:
        errType = 'absMse'
        printf (resFile, f'// cntrSize={cntrSize}, errType={errType}\n')
        myResFileParser = ResFileParser ()
        myResFileParser.printRndErrTableRow (
            distStrs = ['Resnet18', 'Resnet50', 'MobileNet_V2', 'MobileNet_V3'],
            # distStrs = ['uniform', 'norm', 't_5', 't_8', 'Resnet18', 'Resnet50', 'MobileNet_V2', 'MobileNet_V3'],
            cntrSize = cntrSize,
            resFile  = resFile,
            errType  = errType,
            )
        printf (resFile, '\n')

def printAllOptModes ():
    """
    Print the optimal modes for all the given modes.
    Find in the .pcl files the mode (e.g., FP_2e, F2P_li_h2) that minimizes the error for the given distribution.
    """
    resFile = open ('../res/allOptModes.res', 'a')
    for cntrSize in [8, 16, 19]:
        myResFileParser = ResFileParser ()
        # for errType in ['relMse']:
        errType = 'absMse'
        printf (resFile, f'// cntrSize={cntrSize}, errType={errType}\n')
        for distStr in ['uniform', 'norm', 't_5', 't_8', 't_2', 't_4', 't_6', 't_10']: #'MobileNet_V2', 'MobileNet_V3', 'Resnet18', 'Resnet50']:#, 'uniform', 'norm', 't_5', 't_8', 't_2', 't_4', 't_6', 't_10']: 
            bestF2PPoint    = myResFileParser.optModeOfDist (cntrSize=cntrSize, distStr=distStr, errType=errType, onlyF2P=True,  onlyNonF2P=False)
            bestNonF2PPoint = myResFileParser.optModeOfDist (cntrSize=cntrSize, distStr=distStr, errType=errType, onlyF2P=False, onlyNonF2P=True)
            if bestF2PPoint==None or bestNonF2PPoint==None:
                continue
            printf (resFile, f'distStr={distStr}\t bestNonF2P={bestNonF2PPoint[0]}\t, bestNonF2PVal={bestNonF2PPoint[1]}\tbestF2PFlavor={bestF2PPoint[0]}\t, bestF2P/bestNonF2P={bestF2PPoint[1]/bestNonF2PPoint[1]}\n')
        if cntrSize in [8, 19]:
            continue
        elif cntrSize==16:
            ignoreModes=['FP_e10']
        printf (resFile, f'// cntrSize={cntrSize}, errType={errType}\n')
        printf (resFile, f'// Ignoring modes {ignoreModes}\n')
        for distStr in ['MobileNet_V2', 'MobileNet_V3', 'Resnet18', 'Resnet50']:#, 'uniform', 'norm', 't_5', 't_8', 't_2', 't_4', 't_6', 't_10']: 
            bestF2PPoint    = myResFileParser.optModeOfDist (cntrSize=cntrSize, distStr=distStr, errType=errType, onlyF2P=True,  onlyNonF2P=False)
            bestNonF2PPoint = myResFileParser.optModeOfDist (cntrSize=cntrSize, distStr=distStr, errType=errType, onlyF2P=False, onlyNonF2P=True, ignoreModes=ignoreModes)
            if bestF2PPoint==None or bestNonF2PPoint==None:
                continue
            printf (resFile, f'distStr={distStr}\t bestNonF2P={bestNonF2PPoint[0]}\t, bestNonF2PVal={bestNonF2PPoint[1]}\tbestF2PFlavor={bestF2PPoint[0]}\t, bestF2P/bestNonF2P={bestF2PPoint[1]/bestNonF2PPoint[1]}\n')   
        printf (resFile, f'\n')

def plotErVsCntrSize (): 
    """
    Plot the error as a function of the counter's size.
    """
    my_ResFileParser = ResFileParser ()
    erTypes = ['RdMse'] #, 'WrRmse']
    abs = False
    for erType in erTypes: #'WrEr', 'WrRmse', 'RdEr', 'RdRmse', 
        my_ResFileParser.rdPcl (pclFileName='{}_1cntr_HPC_{}.pcl' .format ('abs' if abs else 'rel', erType))
        my_ResFileParser.genErVsCntrSizePlot(erType=erType, numOfExps=100, maxCntrSize=16) 


def genErVsCntrSizeSingleCntr ():
    """
    Generate a table showing the error as a function of the counter's size.
    """
    my_ResFileParser = ResFileParser ()
    outputFileName = f'1cntr.dat' 
    datOutputFile = open (f'../res/{outputFileName}', 'a+')
    abs     = True
    my_ResFileParser.rdPcl (pclFileName='1cntr_PC.pcl', exitError=True)
    maxValBy = 'F3P_li_h3'
    for rel_abs_n in [True, False]:
        for statType in ['normRmse']:
            printf (datOutputFile, '\n// {} {}\n' .format ('rel' if rel_abs_n else 'abs', statType))
            my_ResFileParser.genErVsCntrSizeTable(
                modes           = ['F3P_li_h3', 'CEDAR', 'Morris', 'SEAD_dyn'],
                datOutputFile   = datOutputFile, 
                numOfExps       = 100, 
                cntrSizes       = [8, 10],
                statType        = statType,
                maxValBy        = maxValBy,
                rel_abs_n       = rel_abs_n,
                normalizeByPerfectCntr  = False,
            ) 

def genErVsCntrSizeTableTrace ():
    """
    Generate a table showing the error as a function of the counter's size.
    """
    my_ResFileParser    = ResFileParser ()
    datOutputFile       = open (f'../res/cms_Caida1.dat', 'a+')
    for mode in ['F2P_li_h2', 'F3P_li_h2', 'F3P_li_h3', 'F3P_si_h2', 'F3P_si_h3']:
        # my_ResFileParser.rdPcl (pclFileName=f'cms_{mode}_PC.pcl')
        my_ResFileParser.rdPcl (pclFileName=f'cms_{mode}_HPC_u.pcl')
    for rel_abs_n in [False]:
        for width in [2**i for i in range (8, 19)]:
            for statType in ['normRmse']:
                printf (datOutputFile, '\n// width={} {} {}\n' .format (width, 'rel' if rel_abs_n else 'abs', statType))
                my_ResFileParser.genErVsCntrSizeTable(
                    datOutputFile   = datOutputFile, 
                    numOfExps       = 10, 
                    cntrSizes       = [8],
                    statType        = statType,
                    rel_abs_n       = rel_abs_n,
                    width           = width, 
                    modes           = ['F2P_li_h2', 'F3P_li_h2', 'F3P_li_h3', 'CEDAR', 'Morris', 'SEAD_dyn'],
                    normalizeByPerfectCntr  = False,
                    normalizeByMinimal      = False
                ) 

def rmvFromPcl ():
    myResFileParser = ResFileParser()
    for pclFileName in ['cms_Caida1_PC.pcl', 'cms_Caida2_PC.pcl']:
        myResFileParser.rmvFromPcl(
            pclFileName = pclFileName,
            listOfDicts = [
                {'mode' : 'AEE'}
            ]
        )
        
def uniqListOfDicts (L):
    """
    Given a list of dicts L, returns a list identical to L, but by removing duplicates.
    """ 
    return list({str(i):i for i in L}.values())


def genUniqPcl (
        pclFileName : str
        ):
    """
    Given a fileName.pcl, write to fileName_u.pcl all the pickled items, but by removing duplications. 
    """
    
    myResFileParser = ResFileParser ()
    myResFileParser.rdPcl (pclFileName)
    pclFileNameWoExtension = pclFileName.split('.pcl')[0]
    points = uniqListOfDicts (myResFileParser.points)    
    pclOutputFile = open (f'../res/pcl_files/{pclFileNameWoExtension}_u.pcl', 'wb+')
    for point in points:
        pickle.dump(point, pclOutputFile) 

def genErVsMemSizePlotCms (
        maxValByStr : str, # the mode by which the maximum value of a counter is set.
        ignoreModes : list = [],# List of modes to NOT include in the plot
    ):
    """
    Read the relevant .pcl files, and generate plots showing the error as a function of the overall memory size.
    This function is used to show the results of CMS (Count Min Sketch) simulations.        
    """
    for traceName in ['Caida1', 'Caida2']:
        myResFileParser = ResFileParser ()
        myResFileParser.rdPcl (pclFileName=f'cms_{traceName}_PC_by_{maxValByStr}.pcl')
        myResFileParser.rdPcl (pclFileName=f'cms_{traceName}_HPC_by_{maxValByStr}.pcl')
        myResFileParser.genErVsMemSizePlotCms (
            traceName   = traceName,
            ignoreModes = ignoreModes,
            rel_abs_n   = True,
            maxValByStr = maxValByStr,
            cntrSize    = 8,
        )

def genErVsMemSizePlotSpaceSaving (
        ignoreModes : list = [],# List of modes to NOT include in the plot
    ):
    """
    Read the relevant .pcl files, and generate plots showing the error as a function of the overall memory size.
    This function is used to show the results of Space Saving simulations.        
    """
    for traceName in ['Caida1', 'Caida2']:
        myResFileParser = ResFileParser ()
        myResFileParser.rdPcl (pclFileName=f'ss_{traceName}_PC.pcl')
        # myResFileParser.rdPcl (pclFileName=f'ss_{traceName}_HPC.pcl')
        myResFileParser.genErVsMemSizePlotSpaceSaving (
            traceName   = traceName,
            ignoreModes = ignoreModes,
            rel_abs_n   = False,
            cntrSize    = 8,
        )


def rmvDuplicatedPoints (
    ):
    """
    Remove duplicated points from the results. 
    Save the results while leaving only a single example of each setting.
    """
    myResFileParser = ResFileParser()
    myResFileParser.rmvDuplicatedPoints (
        pclFileName = 'cms_Caida2_HPC_by_None.pcl',
    )

if __name__ == '__main__':
    try:
        # rmvDuplicatedPoints ()
        # genResolutionPlot ()
        genErVsMemSizePlotCms (
            maxValByStr = 'None',
            ignoreModes = []#, 'SEAD_stat_e3', 'SEAD_stat_e4', 'F2P_li_h2'] #, 'F3P_li_h3']
        )
        # genUniqPcl (pclFileName='cms_Caida2_PC.pcl')
        # genErVsCntrSizeSingleCntr ()
        # genErVsCntrSizeTableTrace ()
        # plotErVsCntrSize ()
        # rmvFromPcl ()
        # genQuantErrTable ()
    except KeyboardInterrupt:
        print('Keyboard interrupt.')
