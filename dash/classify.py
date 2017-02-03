import os
import sys
import numpy as np
from dash.download_data_files import download_all_files

scriptDirectory = os.path.dirname(os.path.abspath(__file__))

from dash.restore_model import LoadInputSpectra, BestTypesListSingleRedshift, get_training_parameters

try:
    from PyQt4 import QtGui
    from dash.gui_main import MainApp
except ImportError:
    print("Warning: You will need to install 'PyQt4' if you want to use the graphical interface. " \
          "Using the automatic library will continue to work.")



class Classify(object):
    def __init__(self, filenames=[], redshifts=[], smooth=15):
        """ Takes a list of filenames and corresponding redshifts for supernovae.
        Files should contain a single spectrum, and redshifts should be a list of corresponding redshift floats
        """
        self.filenames = filenames
        self.redshifts = redshifts
        self.smooth = smooth
        self.numSpectra = len(filenames)
        self.mainDirectory = os.path.dirname(os.path.abspath(__file__))

        download_all_files()

        self.modelFilename = os.path.join(self.mainDirectory, "model_trainedAtZeroZ.ckpt")

    def _get_images(self, filename, redshift, trainParams):
        nTypes, w0, w1, nw, minAge, maxAge, ageBinSize, typeList = trainParams
        loadInputSpectra = LoadInputSpectra(filename, redshift, redshift, self.smooth, trainParams)
        inputImage, inputRedshift, typeNamesList, nw, nBins = loadInputSpectra.input_spectra()

        return inputImage, typeNamesList, nw, nBins

    def _input_spectra_info(self):
        trainParams = get_training_parameters()
        nTypes, w0, w1, nw, minAge, maxAge, ageBinSize, typeList = trainParams
        inputImages = np.empty((0, int(nw)), np.float32)
        for i in range(self.numSpectra):
            f = self.filenames[i]
            z = self.redshifts[i]
            inputImage, typeNamesList, nw, nBins = self._get_images(f, z, trainParams)
            inputImages = np.append(inputImages, inputImage, axis=0)
        bestTypesList = BestTypesListSingleRedshift(self.modelFilename, inputImages, typeNamesList, nw, nBins)
        bestTypes = bestTypesList.bestTypes
        softmaxes = bestTypesList.softmaxOrdered

        return bestTypes, softmaxes

    def list_best_matches(self, n=1):
        """Returns a list of lists of the the top n best matches for each spectrum"""
        bestTypes, softmaxes = self._input_spectra_info()
        bestMatchLists = []
        for specNum in range(self.numSpectra):
            bestMatchList = []
            for i in range(n):
                name, age = bestTypes[specNum][i].split(': ')
                prob = softmaxes[specNum][i]
                bestMatchList.append((name, age, prob))
            bestMatchList = np.array(bestMatchList)
            bestMatchLists.append(bestMatchList)
        bestMatchLists = np.array(bestMatchLists)
        # self.bestMatchLists = bestMatchLists
        # self.n = n

        return bestMatchLists

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
        form.fit_spectra()
        form.show()
        app.exec_()






# # EXAMPLE USAGE:
# classification = Classify(filenames=['/Users/dmuthukrishna/Users/dmuthukrishna/DES16E1dic_E1_combined_161125_v10_b00.dat',
#                                      '/Users/dmuthukrishna/Users/dmuthukrishna/DES16E1dic_E1_combined_161125_v10_b00.dat'],
#                           redshifts=[0.34, 0.13])
# print(classification.list_best_matches())




