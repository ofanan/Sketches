import matplotlib, seaborn, pickle, os
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.pylab as pylab
import os, numpy as np, pandas as pd
from pandas._libs.tslibs import period
from printf import printf, printFigToPdf 
from nltk.corpus.reader import lin

import settings
from settings import warning, error, VERBOSE_RES, VERBOSE_PCL, getFxpSettings

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

# Color-blind friendly pallette
BLACK       = '#000000' 
ORANGE      = '#E69F00'
SKY_BLUE    = '#56B4E9'
GREEN       = '#009E73'
YELLOW      = '#F0E442'
BLUE        = '#0072B2'
VERMILION   = '#D55E00'
PURPLE      = '#CC79A7'

# The colors used for each alg's plot, in the dist' case
colorOfMode = {
    'F3P'       : PURPLE,
    'SEAD stat' : VERMILION,
    'SEAD dyn'  : VERMILION,
    'FP'        : BLUE,
    'Tetra stat': BLUE,
    'Tetra dyn' : BLUE,
    'CEDAR'     : BLUE,
    'Morris'    : ORANGE,
    'AEE'       : YELLOW,
    'int'       : 'black',
    'F2P lr'    : GREEN,
    'F2P lr h1' : GREEN,
    'F2P lr h2' : GREEN,
    'F2P_lr_h2' : GREEN,
    'F2P_li_h2' : GREEN,
    'F2P_li'    : GREEN,
    'F2P sr'    : PURPLE,
    'F2P sr h1' : PURPLE,
    'F2P sr h2' : PURPLE,
    'F2P_sr_h2' : PURPLE,
    'F2P_si_h2' : PURPLE,
    'F2P sr'    : PURPLE,
    'F2P li'    : YELLOW,
    'FP 5M2E'   : VERMILION,
    'FP_e2'     : VERMILION,
    'FP_e5'     : BLUE,
    'FP 1M6E'   : BLUE,
    }

colors = ['green', 'purple', 'brown', 'black', 'blue', 'yellow', 'magenta', 'red', 'green', 'purple', 'brown', 'black']

# The markers used for each alg', in the dist' case
markerOfMode = {'F2P_li'    : 'o',
                     'F2P_lr'    : 'o',
                     'F2P_sr'    : 'o',
                     'F3P'       : 'v',
                     'SEAD stat' : '^',
                     'SEAD dyn'  : 's',
                     'FP'        : 'p',
                     'Tetra stat': 'p',
                     'Tetra dyn' : 'X',
                     'CEDAR'     : '<',
                     'Morris'    : '>',
                     'AEE'       : 'o'}

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
    settings.error (f'In ResFileParer.labelOfMode(). Sorry, the mode {mode} is not supported')

def genFpLabel (mantSize : int, expSize : int) -> str:
    """
    Generates a label string that details the counter's settings (param vals), to be used in plots.
    """
    return f'FP {mantSize}M{expSize}E'

# def genFxpLabel (
#         nSystem     : str, #either 'F2P', or 'F3P' 
#         flavor      : str, # flavor, e.g., 'lr', 'sr', 'li'
#         hyperSize   : int = 2 
#     ) -> str:
#     """
#     Given the parameters of F2P or F3P, generates a label string that details the counter's settings (param vals), to be used in plots.
#     """
#     return f'{nSystem} {flavor} h{hyperSize}'

def genFxpLabel (mode : str): # a mode describing the mode flavors
    """
    Given a string that details the parameters of F2P or F3P, generate a label string to be used in plots.
    """
    labelOfMode = {
    'F2P_lr_h2' : r'F2P$_{LR}^2$',
    'F2P_sr_h2' : r'F2P$_{SR}^2$',
    'F2P_li_h2' : r'F2P$_{LI}^2$',
    'F2P_si_h2' : r'F2P$_{SI}^2$',
    'F3P_lr_h2' : r'F3P$_{LR}^2$',
    'F3P_sr_h2' : r'F3P$_{SR}^2$',
    'F3P_li_h2' : r'F3P$_{LI}^2$',
    'F3P_si_h2' : r'F3P$_{SI}^2$',
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
        'font.size': FONT_SIZE,
        'legend.fontsize': LEGEND_FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE, }) if (size == 'large') else matplotlib.rcParams.update({
        'font.size': FONT_SIZE_SMALL,
        'legend.fontsize': LEGEND_FONT_SIZE_SMALL,
        'xtick.labelsize': FONT_SIZE_SMALL,
        'ytick.labelsize': FONT_SIZE_SMALL,
        'axes.labelsize': FONT_SIZE_SMALL,
        'axes.titlesize':FONT_SIZE_SMALL
        })
    
    def __init__ (self):
        """
        Initialize a Res_file_parser, used to parse result files, and generate plots. 
        """
        # List of algorithms' names, used in the plots' legend, for the dist' case
        self.markers = ['o', 'v', '^', 's', 'p', 'X']
        self.points = []
        
    def rdPcl (self, pclFileName):
        """
        Given a .pcl filename, read all the data it contains into self.points
        """
        settings.checkIfInputFileExists ('../res/pcl_files/{}' .format(pclFileName))
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
            erType         = 'RdRmse',
            numOfExps      = 50,
            modes          = ['F2P_li', 'CEDAR', 'Morris'], # 'SEAD_dyn'],
            minCntrSize    = 8,
            maxCntrSize    = 64,
        ):
        """
        Generate a plot showing the error as a function of the counter's size.
        """

        outputFileName = f'1cntr_PC_{erType}' 
        self.setPltParams ()  # set the plot's parameters (formats of lines, markers, legends etc.).
        _, ax = plt.subplots()

        for mode in modes:
            pointsOfThisMode = [point for point in self.points if point['mode'] == mode and point['numOfExps'] == numOfExps and point['erType'] == erType]
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
                ax.plot ((cntrSize, cntrSize), (y_lo, y_hi), color=colorOfMode[mode])  # Plot the conf' interval line
                y.append (y_avg)
            ax.plot (cntrSizes, y, color=colorOfMode[mode], marker=markerOfMode[mode],
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
            modes       : list = ['F2P_li', 'CEDAR', 'Morris', 'SEAD_dyn'],
            cntrSizes   : list = [],
            statType    : str  = 'Mse',
            rel_abs_n   : bool = False, # When True, consider relative errors, Else, consider absolute errors.
            width       : int  = None,  # The width of the CMS. Relevant only for CMS' sim results.
            normalizeByPerfectCntr : bool = True, # when True, normalize all mode's results by dividing them by the value obtained by a perfect counter
            normalizeByMinimal     : bool = True # when True, normalize all mode's results by dividing them by the value obtained by the lowest value at this row
        ):
        """
        Generate a table showing the error as a function of the counter's size.
        """
    
        points = [point for point in self.points if point['numOfExps'] == numOfExps and point['statType']==statType and point['rel_abs_n']==rel_abs_n]
        if width!=None:
            points = [point for point in points if point['width']==width]
    
        printf (datOutputFile, '\t')
        for mode in modes:
            printf (datOutputFile, f'{mode}\t')
            if mode!=modes[-1]:
                printf (datOutputFile, '& ')
        printf (datOutputFile, ' \\\\ \n')
        for cntrSize in cntrSizes:
            pointsOfThisCntrSize = [point for point in points if point['cntrSize']==cntrSize]
            printf (datOutputFile, f'{cntrSize} & ')
            pointsOfThisCntrSizeErType = [point for point in pointsOfThisCntrSize]  
            if pointsOfThisCntrSizeErType == []:
                error (f'No points found for numOfExps={numOfExps}, cntrSize={cntrSize}')
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
                    warning (f'No points found for numOfExps={numOfExps}, cntrSize={cntrSize}, mode={mode}')
                    if mode!=modes[-1]:
                        printf (datOutputFile, ' & ')
                    continue
                if len(pointsToPrint)>1:
                    warning (f'found {len(pointsToPrint)} points for numOfExps={numOfExps}, cntrSize={cntrSize}, mode={mode}')                
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
    
    def genErVsCntrMaxValPlot (self, cntrSize=8, plotAbsEr=True):
        """
        Generate a plot showing the relative / abs err as a function of the maximum counted value
        Inputs:
            cntrSize - size of the compared counters.
            plotAbsEr - if True, plot the absolute errors. Else, plot the relative errors
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

            ax.plot (cntrMaxVals, y, color=colorOfMode[mode], marker=markerOfMode[mode],
                     markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=point['settingStr'], mfc='none') 

        plt.xlabel('Counter Maximum Value')
        plt.ylabel('Avg. {} Eror' .format ('Absolute' if plotAbsEr else 'Relative'))
        ax.set_xscale ('log')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend (by_label.values(), by_label.keys(), fontsize=LEGEND_FONT_SIZE) #, frameon=False)        
        plt.savefig ('../res/{}.pdf' .format (outputFileName), bbox_inches='tight')        
        
       
    def genResolutionPlotByModes (self,
            modes           = [],   # modes for which the plot will be generated. 
            cntrSize        = 8,    # cntrSizes for which the plot will be generated. 
            minCntrVal      = 0,  # min' X (counter) value at the plot
            maxCntrVal      = float('inf'), # max X (counter) value at the plot
            xLog            = False,    # When True, plot the x axis in a log' scaling.
            ) -> None:
        """
        Generate a plot showing the resolution as a function of the counted val for the given modes
        """
        self.setPltParams   ()  # set the plot's parameters (formats of lines, markers, legends etc.).
        _, ax = plt.subplots()
        for mode in modes:
            pointsOfThisCntrSize = [point for point in self.points if point['cntrSize'] == cntrSize]
            pointsOfThisMode = [point for point in pointsOfThisCntrSize if point['mode'] == mode]
            if pointsOfThisMode == []:
                print (f'No points found for mode {mode}')
                continue
            if len(pointsOfThisMode) < 1:
                settings.error (f'More than a single list of points for mode {mode}')
            points = pointsOfThisMode[0]['points']
            
            ax.plot (points['X'], points['Y'], color=colorOfMode[mode], marker=markerOfMode[mode],
                     markersize=MARKER_SIZE_SMALL, linewidth=LINE_WIDTH_SMALL, label=mode, mfc='none') 

        plt.xlabel('Counted Value')
        plt.ylabel(f'Relative Resolution')
        plt.yscale ('log')
        if xLog: 
            plt.xscale ('log')

        conf        = settings.getConfByCntrSize (cntrSize=cntrSize)
        plt.xlim ([minCntrVal, conf['cntrMaxVal']+1])
        # plt.ylim (0.01, 0.1) 
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend (by_label.values(), by_label.keys(), fontsize=LEGEND_FONT_SIZE, frameon=False)        
        plt.savefig ('../res/ResolutionByModes_n{}_{}.pdf' .format (cntrSize, 'log' if xLog else 'lin'), bbox_inches='tight')        


    def genResolutionPlotBySettingStrs (self,
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
                     markersize=MARKER_SIZE_SMALL, linewidth=LINE_WIDTH_SMALL, label=settingStr, mfc='none')
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
    """
    
    my_ResFileParser = ResFileParser ()
    byModes = False
    if byModes:
        my_ResFileParser.rdPcl (pclFileName=f'resolutionByModes.pcl')
        for cntrSize in [8]:  # , 12, 16]:
            my_ResFileParser.genResolutionPlotByModes (
                modes       = ['SEAD stat', 'Morris', 'F2P_li'], #  
                minCntrVal  = 1000,
                maxCntrVal  = float('inf'),
                cntrSize    = cntrSize,
                xLog        = False
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
    
def genRndErrTable ():
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
        my_ResFileParser.rdPcl (pclFileName='1cntr_PC.pcl')
        printf (datOutputFile, '\n// {}\n' .format ('abs ' if abs else 'rel '))
        for rel_abs_n in [True, False]:
            for statType in ['Mse', 'normRmse']:
                printf (datOutputFile, '\n// {} {}\n' .format ('rel' if rel_abs_n else 'abs', statType))
                my_ResFileParser.genErVsCntrSizeTable(
                    modes           = ['F2P_li', 'CEDAR', 'Morris', 'SEAD_dyn'],
                    datOutputFile   = datOutputFile, 
                    numOfExps       = 100, 
                    cntrSizes       = [8, 10, 12, 14, 16],
                    statType        = statType,
                    rel_abs_n       = rel_abs_n,
                    normalizeByPerfectCntr = False
                ) 

def genErVsCntrSizeTableTrace ():
        """
        Generate a table showing the error as a function of the counter's size.
        """
        my_ResFileParser = ResFileParser ()
        outputFileName = f'cms.dat' 
        datOutputFile = open (f'../res/{outputFileName}', 'a+')
        my_ResFileParser.rdPcl (pclFileName='cms_li_PC.pcl')
        width = 2**12
        for rel_abs_n in [False]:
            for statType in ['Mse']:
                printf (datOutputFile, '\n// width={} {} {}\n' .format (width, 'rel' if rel_abs_n else 'abs', statType))
                my_ResFileParser.genErVsCntrSizeTable(
                    datOutputFile   = datOutputFile, 
                    numOfExps       = 2, 
                    cntrSizes       = [8], #, 10, 12, 14, 16],
                    statType        = statType,
                    rel_abs_n       = rel_abs_n,
                    width           = width, 
                    normalizeByPerfectCntr = False
                ) 

def rmvFromPcl ():
    myResFileParser = ResFileParser()
    myResFileParser.rmvFromPcl(
        pclFileName = 'rndErr_n8.pcl',
        listOfDicts = [
            {'mode' : 'F3P_lr_h1'},
            {'mode' : 'F3P_lr_h2'},
            {'mode' : 'F3P_lr_h3'},
            {'mode' : 'F3P_sr_h1'},
            {'mode' : 'F3P_sr_h2'},
            {'mode' : 'F3P_sr_h3'}]
        )
        
if __name__ == '__main__':
    try:
        # genErVsCntrSizeSingleCntr ()
        # genErVsCntrSizeTableTrace ()
        # plotErVsCntrSize ()
        # rmvFromPcl ()
        genRndErrTable ()
    except KeyboardInterrupt:
        print('Keyboard interrupt.')

# genResolutionPlot ()
    # my_ResFileParser.printAllPoints (cntrSize=8, cntrMaxVal=1488888, printToScreen=True)

        #     verbose     = [VERBOSE_RES]
        #     )
        # plotErVsCntrSize ()
        # printAllOptModes ()
        # calcOptModeByDist ()
        # genErrByDistBar ()
        # genErrByDfGraph ()
