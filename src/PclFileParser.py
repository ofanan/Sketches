import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import pickle

MARKER_SIZE            = 16
MARKER_SIZE_SMALL      = 1
LINE_WIDTH             = 3
LINE_WIDTH_SMALL       = 1
FONT_SIZE              = 20
FONT_SIZE_SMALL        = 5
LEGEND_FONT_SIZE       = 14
LEGEND_FONT_SIZE_SMALL = 5


class PclFileParser(object):
    """
    Parse pcl files, and generate plots from them.
    """

    # Set the parameters of the plot (sizes of fonts, legend, ticks etc.).
    # mfc='none' makes the markers empty.
    setPltParams = lambda self, size='large': matplotlib.rcParams.update({'font.size'      : FONT_SIZE,
                                                                          'legend.fontsize': LEGEND_FONT_SIZE,
                                                                          'xtick.labelsize': FONT_SIZE,
                                                                          'ytick.labelsize': FONT_SIZE,
                                                                          'axes.labelsize' : FONT_SIZE,
                                                                          'axes.titlesize' : FONT_SIZE, }) if (
            size == 'large') else matplotlib.rcParams.update({
        'font.size'      : FONT_SIZE_SMALL,
        'legend.fontsize': LEGEND_FONT_SIZE_SMALL,
        'xtick.labelsize': FONT_SIZE_SMALL,
        'ytick.labelsize': FONT_SIZE_SMALL,
        'axes.labelsize' : FONT_SIZE_SMALL,
        'axes.titlesize' : FONT_SIZE_SMALL
    })

    def __init__(self):
        """
        Initialize a pcl_file_parser, used to parse result files, and generate plots.
        """
        # List of algorithms' names, used in the plots' legend, for the dist' case
        self.labelOfMode = {}

        # The colors used for each alg's plot, in the dist' case
        self.colorOfMode = {'F2P'         : 'green',
                             'realCounter': 'blue',
                            'CEDAR'       : 'brown',
                            'Morris'      : 'red'}

        # The markers used for each alg', in the dist' case
        self.markerOfMode = {'F2P'         : 'o',
                              'realCounter': 'v',
                             'CEDAR'       : '<',
                             'Morris'      : '>'}
        self.points      = []

    def rdPcl(self, pclFileName):
        """
        Given a RdRmse.pcl, read all the data it contains into self.points
        """
        pclFile = open('../res/pcl_files/{}.pcl' .format(pclFileName), 'rb')

        while True:
            try:
                self.points.append(pickle.load(pclFile))
            except EOFError:
                break

    def NRMSEVsWidthPlot(self, modes, pclFileName):
        """
        Generate a plot showing the Normalized_RMSE vs. width.
        """

        self.setPltParams()  # set the plot's parameters (formats of lines, markers, legends etc.).
        _, ax = plt.subplots()
        for mode in modes:
            pointsOfThisMode = [point for point in self.points if point['mode'] == mode]
            if pointsOfThisMode == []:
                print(f'No points found for mode {mode}')
                continue
            numCntrs = [point['numCntrs'] for point in pointsOfThisMode]
            y_lo = [point['Lo'] for point in pointsOfThisMode]
            y_avg=[point['Avg'] for point in pointsOfThisMode]
            y_hi = [point['Hi'] for point in pointsOfThisMode]
            ax.plot((numCntrs, numCntrs), (y_lo, y_hi), color=self.colorOfMode[mode])  # Plot the conf' interval line
            ax.plot(numCntrs, y_avg, color=self.colorOfMode[mode], marker=self.markerOfMode[mode],
                     markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=mode, mfc='none')

        plt.xlabel('numCntrs')
        plt.ylabel('AvgRdError')
        plt.title('AvgRdError vs. numCntrs')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=LEGEND_FONT_SIZE)
        plt.savefig('../res/{}.pdf'.format(pclFileName), bbox_inches='tight')
