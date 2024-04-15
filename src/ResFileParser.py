import matplotlib, seaborn
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.pylab as pylab
import numpy as np, pandas as pd
from pandas._libs.tslibs import period
from printf import printf, printFigToPdf 
import pickle
import settings
from nltk.corpus.reader import lin

MARKER_SIZE = 16
MARKER_SIZE_SMALL = 1
LINE_WIDTH = 3 
LINE_WIDTH_SMALL = 2 
FONT_SIZE = 20
FONT_SIZE_SMALL = 5
LEGEND_FONT_SIZE = 14
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

colors = ['green', 'purple', 'brown', 'black', 'blue', 'yellow', 'magenta', 'red', 'green', 'purple', 'brown', 'black']

# The colors used for each alg's plot, in the dist' case
colorOfMode = {
    'F3P'       : 'purple',
    'SEAD stat' : 'brown',
    'SEAD dyn'  : 'yellow',
    'FP'        : 'blue',
    'Tetra stat': 'blue',
    'Tetra dyn' : 'black',
    'CEDAR'     : 'magenta',
    'Morris'    : 'red',
    'AEE'       : 'blue',
    'int'       : 'black',
    'F2P lr'    : 'green',
    'F2P lr h1' : 'green',
    'F2P lr h2' : 'green',
    'F2P sr'    : 'purple',
    'F2P sr h1' : 'purple',
    'F2P sr h2' : 'purple',
    'F2P sr'    : 'purple',
    'F2P li'    : 'yellow',
    'FP 5M2E'   : 'magenta',
    'FP 1M6E'   : 'blue',
    }

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

def genRndErrFileName (cntrSize : int) -> str:
    """
    Given the counter's size, generate the .pcl filename.
    """
    return f'rndErr_n{cntrSize}'

def genFpLabel (mantSize : int, expSize : int) -> str:
    """
    Generates a label string that details the counter's settings (param vals), to be used in plots.
    """
    return f'FP {mantSize}M{expSize}E'

def genF2pLabel (flavor    : str, # flavor, e.g., 'lr', 'sr', 'li'
                 hyperSize : int = 2 
                 ) -> str:
    """
    Generates a label string that details the counter's settings (param vals), to be used in plots.
    """
    return f'F2P {flavor} h{hyperSize}'

def f2pSettingsToLabel (mode : str) -> str:
    """
    Given a string detailing F2P's settings, return the corresponding label
    """
    F2pSettings = getF2PSettings(mode)
    return genF2pLabel (flavor = F2pSettings['flavor'], hyperSize=F2pSettings['hyperSize'])

def getF2PSettings (mode : str) -> dict:
    """
    given the mode string of an F2P counter, get a dictionary detailing its settings (flavor and hyperExp size).
    """
    return {'flavor'    : mode.split('F2P_')[1].split('_')[0],
            'hyperSize' : int(mode.split('_h')[1].split('_')[0])}


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
        self.labelOfMode = {}
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
            
    # def addToPcl (self, erType, pclOutputFileName):
    #     """
    #     Dump all the points in self.points to a .pcl. file
    #     """
    #
    #     pclOutputFile = open(f'../res/pcl_files/{pclOutputFileName}', 'ab+')
    #
    #     for point in self.points:
    #         point['erType'] = erType
    #         pickle.dump(point, pclOutputFile) 

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

    def genErVsCntrSizePlot (self,
                             erType,
                             numOfExps      = 50,
                             modes          = ['F2P_li', 'CEDAR', 'Morris'],
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
                     markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=point['mode'], mfc='none') 

        plt.xlabel('Counter Size [bits]')
        plt.ylabel('RMSE')
        plt.yscale ('log')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend (by_label.values(), by_label.keys(), fontsize=LEGEND_FONT_SIZE, frameon=False)        
        if not(USE_FRAME):
            seaborn.despine(left=True, bottom=True, right=True)
        plt.savefig ('../res/{}.pdf' .format (outputFileName), bbox_inches='tight')        
    
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


    def genMseByDfGraph (self,
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
        
        if settings.VERBOSE_RES in verbose:
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
            yVals             = [point[f'{resTypeStr}Mse'] for point in pointsOfThisMode]
            if settings.VERBOSE_RES in verbose:
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

        if settings.VERBOSE_RES in verbose:
            printf (resFile, '\n')
        if settings.VERBOSE_PLOT not in verbose:
            return
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend (by_label.values(), by_label.keys(), fontsize=10, frameon=False)
        plt.xlim (1, 100)
        plt.xlabel (f'df')
        plt.ylabel (f'{resTypeStr} MSE')
        plt.yscale ('log')
        # plt.show ()
        plt.savefig (f'../res/student_rndErr_n{cntrSize}_{resTypeStr}.pdf', bbox_inches='tight')        

    def MseByDistBar (
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
        
        if settings.VERBOSE_RES in verbose:
            resFile = open (f'../res/{genRndErrFileName(cntrSize)}.res', 'a+')
            printf (resFile, f'// errType={resTypeStr}, dist={dist}\n')
            printedDfs = False

        _, ax = plt.subplots()
        printedDFs = False
        
        if dist.startswith('t'): # For a student distribution, need to pick the points with the desired df
            dist_split = dist.split('_')
            if len(dist_split)<1:
                settings.error ('In ResFileParser.MseByDistBar(): Student dist name should be in the format Student_df')
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
                print (f'In MseByDistBar(): no points for dist Student, df={df}, mode={mode}')
                continue
            yVals[i] = pointsOfThisMode[0][f'{resTypeStr}Mse']
            if settings.VERBOSE_RES in verbose:
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


        if settings.VERBOSE_RES in verbose:
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
            cntrSize    : int = 8,
            errType     : str = 'abs', # Currently, either 'abs' (absolute) or 'rel' (relative
            distStr     : str = 'uniform', 
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
        points = sorted (points, key = lambda item : item[f'{errType}Mse'])
        if len(points)==0:
            print (f'In ResFileParser.optModeOfDist(). No points found for cntrSize={cntrSize}, errType={errType}, dist={distStr}')
            return None
        return ([points[0]['mode'], points[0][f'{errType}Mse']])

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

def genMseByDfGraph ():
    """
    Plot the MSE as a func' of the df value at the Student-t dist'.
    """
    for cntrSize in [8, 16]:
        myResFileParser = ResFileParser ()
        pclFileName = genRndErrFileName (cntrSize) 
        myResFileParser.rdPcl (pclFileName)
        myResFileParser.genMseByDfGraph (cntrSize=cntrSize, resTypeStr='abs', verbose=[settings.VERBOSE_RES])
        myResFileParser.genMseByDfGraph (cntrSize=cntrSize, resTypeStr='rel', verbose=[settings.VERBOSE_RES])

def genMseByDistBar ():
    """
    Generate and save a bar-plot of the Mean Square Error for the various distributions. 
    """
    for cntrSize in [8]: #, 16, 19]:
        myResFileParser = ResFileParser ()
        myResFileParser.rdPcl (f'{genRndErrFileName(cntrSize)}.pcl')
        for dist in ['uniform']: #, 'Gaussian', 't_5', 't_8']:
            for resTypeStr in ['rel']: #, 'abs']: # a string detailing the y value for which the func' will generate a plot
                myResFileParser.genMseByDistBar (
                    stdev       = 1,
                    dist        = dist,
                    cntrSize    = cntrSize,
                    modes       = settings.modesOfCntrSize (cntrSize),
                    resTypeStr  = resTypeStr,
                    numPts      = None,         # num of points in the experiment
                    verbose     =[settings.VERBOSE_RES, settings.VERBOSE_PLOT]
                )
    
def printAllOptModes ():
    """
    Print the optimal modes for all the given modes.
    Find in the .pcl files the mode (e.g., FP_2e, F2P_li_h2) that minimizes the error for the given distribution.
    """
    myResFileParser = ResFileParser ()
    resFile = open ('../res/allOptModes.res', 'w')
    for cntrSize in [8, 16]:
        for errType in ['abs', 'rel']:
            printf (resFile, f'// cntrSize={cntrSize}, errType={errType}\n')
            for distStr in ['Resnet18', 'Resnet50', 'uniform', 'norm', 't_5', 't_8', 't_2', 't_4', 't_6', 't_10', 't_20', ]:
                bestF2PPoint    = myResFileParser.optModeOfDist (cntrSize=cntrSize, distStr=distStr, errType=errType, onlyF2P=True,  onlyNonF2P=False)
                bestNonF2PPoint = myResFileParser.optModeOfDist (cntrSize=cntrSize, distStr=distStr, errType=errType, onlyF2P=False, onlyNonF2P=True)
                if bestF2PPoint==None or bestNonF2PPoint==None:
                    continue
                printf (resFile, f'distStr={distStr}\t bestNonF2P={bestNonF2PPoint[0]}\t, bestNonF2PVal={bestNonF2PPoint[1]}\tbestF2PFlavor={bestF2PPoint[0]}\t, bestF2P/bestNonF2P={bestF2PPoint[1]/bestNonF2PPoint[1]}\n')   
            printf (resFile, f'\n')

if __name__ == '__main__':
    try:
        printAllOptModes ()
        # calcOptModeByDist ()
        # genMseByDistBar ()
        # genMseByDfGraph ()
    except KeyboardInterrupt:
        print('Keyboard interrupt.')

# genResolutionPlot ()
# my_ResFileParser = ResFileParser ()
# for ErType in ['WrRmse', 'RdRmse']: #'WrEr', 'WrRmse', 'RdEr', 'RdRmse', 
#     my_ResFileParser.rdPcl (pclFileName=f'1cntr_PC_{ErType}.pcl')
#     my_ResFileParser.genErVsCntrSizePlot(ErType, numOfExps=1, maxCntrSize=16) # 50
    # my_ResFileParser.printAllPoints (cntrSize=8, cntrMaxVal=1488888, printToScreen=True)

# print ('{:.2e}' .format (0.000056))