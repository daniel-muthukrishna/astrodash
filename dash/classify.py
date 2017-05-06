import os
import sys
import numpy as np
from dash.download_data_files import download_all_files

scriptDirectory = os.path.dirname(os.path.abspath(__file__))

from dash.restore_model import LoadInputSpectra, BestTypesListSingleRedshift, get_training_parameters
from dash.false_positive_rejection import FalsePositiveRejection, combined_prob

try:
    from PyQt5 import QtGui
    from dash.gui_main import MainApp
except ImportError:
    print("Warning: You will need to install 'PyQt5' if you want to use the graphical interface. "
          "Using the automatic library will continue to work.")


class Classify(object):
    loaded = np.load(os.path.join(scriptDirectory, "data_files/templates.npz"))
    templateImages = loaded['templateFluxesAll']
    # templateLabelsIndexes = loaded['templateLabelsAll']

    def __init__(self, filenames=[], redshifts=[], smooth=15, minWave=2500, maxWave=10000, classifyHost=False):
        """ Takes a list of filenames and corresponding redshifts for supernovae.
        Files should contain a single spectrum, and redshifts should be a list of corresponding redshift floats
        """
        self.filenames = filenames
        self.redshifts = redshifts
        self.smooth = smooth
        self.minWave = minWave
        self.maxWave = maxWave
        self.classifyHost = classifyHost
        self.numSpectra = len(filenames)
        self.mainDirectory = os.path.dirname(os.path.abspath(__file__))

        # download_all_files('v01')

        self.modelFilename = os.path.join(self.mainDirectory, "data_files/model_trainedAtZeroZ.ckpt")

    def _get_images(self, filename, redshift, trainParams):
        loadInputSpectra = LoadInputSpectra(filename, redshift, redshift, self.smooth, trainParams, self.minWave, self.maxWave, self.classifyHost)
        inputImage, inputRedshift, typeNamesList, nw, nBins = loadInputSpectra.input_spectra()

        return inputImage, typeNamesList, nw, nBins

    def _input_spectra_info(self):
        pars = get_training_parameters()
        inputImages = np.empty((0, int(pars['nw'])), np.float16)
        for i in range(self.numSpectra):
            f = self.filenames[i]
            z = self.redshifts[i]
            inputImage, typeNamesList, nw, nBins = self._get_images(f, z, pars)
            inputImages = np.append(inputImages, inputImage, axis=0)
        bestTypesList = BestTypesListSingleRedshift(self.modelFilename, inputImages, typeNamesList, nw, nBins)
        bestTypes = bestTypesList.bestTypes
        softmaxes = bestTypesList.softmaxOrdered
        bestLabels = bestTypesList.idx

        return bestTypes, softmaxes, bestLabels, inputImages

    def list_best_matches(self, n=1):
        """Returns a list of lists of the the top n best matches for each spectrum"""
        bestTypes, softmaxes, bestLabels, inputImages = self._input_spectra_info()
        bestMatchLists = []
        bestBroadTypes = []
        rejectionLabels = []
        reliableFlags = []
        for specNum in range(self.numSpectra):
            bestMatchList = []
            for i in range(20):
                classification = bestTypes[specNum][i].split(': ')
                if len(classification) == 2:
                    name, age = classification
                    host = ""
                else:
                    host, name, age = classification
                prob = softmaxes[specNum][i]
                bestMatchList.append((name, age, prob))
            bestMatchList = np.array(bestMatchList)
            bestMatchLists.append(bestMatchList[0:n])
            bestBroadType, reliableFlag = self.best_broad_type(bestMatchList)
            bestBroadTypes.append(bestBroadType)
            reliableFlags.append(reliableFlag)
            rejectionLabels.append(self.false_positive_rejection(bestLabels[specNum][::-1], inputImages[specNum]))

        bestMatchLists = np.array(bestMatchLists)
        # self.bestMatchLists = bestMatchLists
        # self.n = n

        return bestMatchLists, bestBroadTypes, rejectionLabels, reliableFlags

    def best_broad_type(self, bestMatchList):
        host, prevName, bestAge, probTotal, reliableFlag = combined_prob(bestMatchList[0:10])

        return (prevName, bestAge, probTotal), reliableFlag

    def false_positive_rejection(self, bestLabel, inputImage):
        c = bestLabel[0] # best Index
        templateImages = Classify.templateImages[c]
        if templateImages[0].any():
            falsePositiveRejection = FalsePositiveRejection(inputImage, templateImages)
            rejectionLabel = "(chi2=%s, rlap=%s)" % (falsePositiveRejection.rejection_label(), falsePositiveRejection.rejection_label2())
        else:
            rejectionLabel = "(NO_TEMPLATES)"

        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # plt.plot(inputImage)
        # plt.plot(templateImage)
        # plt.show()

        return rejectionLabel

    # def save_best_matches(self, n=1, filename='DASH_matches.txt'):
    #     if ('self.bestMatchLists' not in locals()) or self.n != n:
    #         self.bestMatchLists = self.list_best_matches(n)
    #     np.savetxt(filename, self.bestMatchLists)

    def plot_with_gui(self, indexToPlot=0):
        app = QtGui.QApplication(sys.argv)
        form = MainApp(inputFilename=self.filenames[indexToPlot])
        form.lblInputFilename.setText(self.filenames[indexToPlot])
        form.lineEditKnownZ.setText(str(self.redshifts[indexToPlot]))
        form.lineEditSmooth.setText(str(self.smooth))
        form.fit_spectra(self.classifyHost)
        form.show()
        app.exec_()






# # EXAMPLE USAGE:
# classification = Classify(filenames=['/Users/dmuthukrishna/Users/dmuthukrishna/DES16E1dic_E1_combined_161125_v10_b00.dat',
#                                      '/Users/dmuthukrishna/Users/dmuthukrishna/DES16E1dic_E1_combined_161125_v10_b00.dat'],
#                           redshifts=[0.34, 0.13])
# print(classification.list_best_matches())




