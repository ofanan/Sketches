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
LINE_WIDTH_SMALL = 1 
FONT_SIZE = 20
FONT_SIZE_SMALL = 5
LEGEND_FONT_SIZE = 14
LEGEND_FONT_SIZE_SMALL = 5 
USE_FRAME              = True # When True, plot a "frame" (box) around the plot 


class ResFileParser (object):
    """
    Parse result files, and generate plots from them.
    """

    # Set the parameters of the plot (sizes of fonts, legend, ticks etc.).
    # mfc='none' makes the markers empty.
    setPltParams = lambda self, size = 'large': matplotlib.rcParams.update({'font.size': FONT_SIZE,
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

        # The colors used for each alg's plot, in the dist' case
        self.colorOfMode = {'F2P_li'    : 'green',
                            'F2P_lr'    : 'green',
                            'F2P_ls'    : 'green',
                            'F3P'       : 'purple',
                            'SEAD stat' : 'brown',
                            'SEAD dyn'  : 'yellow',
                            'FP'        : 'blue',
                            'Tetra stat': 'blue',
                            'Tetra dyn' : 'black',
                            'CEDAR'     : 'magenta',
                            'Morris'    : 'red',
                            'AEE'       : 'blue'}

        # The markers used for each alg', in the dist' case
        self.markerOfMode = {'F2P_li'    : 'o',
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
                ax.plot ((cntrSize, cntrSize), (y_lo, y_hi), color=self.colorOfMode[mode])  # Plot the conf' interval line
                y.append (y_avg)
            ax.plot (cntrSizes, y, color=self.colorOfMode[mode], marker=self.markerOfMode[mode],
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
                pointOfThisModeNMaxVal = [point for point in pointOfThisModeNMaxVal if point['settingsStr'] not in ['F2P_n8_h1', 'SEADstat_n8_e1']]  # $$$
                if (len(pointOfThisModeNMaxVal) != 1):
                    print ('bug at genErVsCntrMaxValPlot: pointOfThisModeNMaxVal!=1. Points are')
                    print (pointOfThisModeNMaxVal) 
                    exit ()
                point = pointOfThisModeNMaxVal[0]
                
                if (plotAbsEr):
                    y_lo, y_avg, y_hi = point['absRdErLo'], point['absRdErAvg'], point['absRdErHi']
                else:
                    y_lo, y_avg, y_hi = point['relRdErLo'], point['relRdErAvg'], point['relRdErHi']
                     
                printf (datOutputFile, 'settingsStr={}, mode={}. cntrMaxVal={}, y_lo={:.2f}, y_hi={:.2f}, y_avg={:.2f}\n' .format 
                                       (point['settingsStr'], mode, cntrMaxVal, y_lo, y_hi, y_avg))                    
                y.append (y_avg)

            label = mode
        
            ax.plot (cntrMaxVals, y, color=self.colorOfMode[mode], marker=self.markerOfMode[mode],
                     markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=point['settingsStr'], mfc='none') 

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
            
            ax.plot (points['X'], points['Y'], color=self.colorOfMode[mode], marker=self.markerOfMode[mode],
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
        plt.savefig ('../res/ResolutionByModes_n{}_{}.pdf' .format (cntrSize, cntrSize, 'log' if xLog else 'lin'), bbox_inches='tight')        


    def genResolutionPlotBySettingStrs (self,
            settingsStrs    = [],           # Concrete settings for which the plot will be generated
            minCntrVal      = 0,            # min' X (counter) value at the plot
            maxCntrVal      = float('inf'), # max X (counter) value at the plot
            xLog            = False,        # When True, plot the x axis in a log' scaling.
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
            
            ax.plot (points['X'], points['Y'], color=self.colorOfMode[mode], marker=self.markerOfMode[mode],
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
        plt.savefig ('../res/Resolution_{}_{}bits_{}.pdf' .format ('int' if isInt else 'real', cntrSize, 'log' if xLog else 'lin'), bbox_inches='tight')        

def genResolutionPlot ():
    """
    """
    
    my_ResFileParser = ResFileParser ()
    byModes = True
    # my_ResFileParser.printAllPoints (printToScreen=True) #$$$
    # settings.error ('rega') #$$
    # modes   = ['F2P_lr', 'FP']
    # minCntrVal  = 0
    if byModes:
        my_ResFileParser.rdPcl (pclFileName=f'resolutionByModes.pcl')
        for cntrSize in [8]:  # , 12, 16]:
            my_ResFileParser.genResolutionPlotByModes (
                modes       = ['Morris', 'SEAD stat', 'F2P_li'],
                minCntrVal  = 1000,
                maxCntrVal  = float('inf'),
                cntrSize    = cntrSize,
                xLog        = False
                )

genResolutionPlot ()
# my_ResFileParser = ResFileParser ()
# for ErType in ['WrRmse']: #'WrEr', 'WrRmse', 'RdEr', 'RdRmse', 
#     my_ResFileParser.rdPcl (pclFileName=f'1cntr_PC_{ErType}.pcl')
#     my_ResFileParser.genErVsCntrSizePlot(ErType, numOfExps=1, maxCntrSize=16) # 50
    # my_ResFileParser.printAllPoints (cntrSize=8, cntrMaxVal=1488888, printToScreen=True)

