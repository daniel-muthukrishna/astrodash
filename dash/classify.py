import os
import sys
import numpy as np
import urllib

scriptDirectory = os.path.dirname(os.path.abspath(__file__))

from restore_model import LoadInputSpectra, BestTypesListSingleRedshift

try:
    from PyQt4 import QtGui
    from main import MainApp
except ImportError:
    print "Warning: You will need to install 'PyQt4' if you want to use the graphical interface. " \
          "Using the automatic library will continue to work."



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

        modelFilename = os.path.join(scriptDirectory, 'model_trainedAtZeroZ.ckpt')
        if not os.path.isfile(modelFilename):
            print "Downloading Training Model..."
            modelFileDownload = urllib.URLopener()
            modelFileDownload.retrieve(
                "https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/model_trainedAtZeroZ.ckpt",
                modelFilename)
            print modelFilename
        dataFilename = os.path.join(scriptDirectory, 'type_age_atRedshiftZero.npz')
        if not os.path.isfile(dataFilename):
            print "Downloading Data File..."
            dataFileDownload = urllib.URLopener()
            dataFileDownload.retrieve(
                "https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/type_age_atRedshiftZero.npz",
                dataFilename)
            print dataFilename
        dataFilename = os.path.join(scriptDirectory, 'training_params.pickle')
        if not os.path.isfile(dataFilename):
            print "Downloading Data File..."
            dataFileDownload = urllib.URLopener()
            dataFileDownload.retrieve(
                "https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/training_params.pickle",
                dataFilename)
            print dataFilename
        dataFilename = os.path.join(scriptDirectory, 'templates.npz')
        if not os.path.isfile(dataFilename):
            print "Downloading Data File..."
            dataFileDownload = urllib.URLopener()
            dataFileDownload.retrieve(
                "https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/templates.npz", dataFilename)
            print dataFilename

        self.modelFilename = os.path.join(self.mainDirectory, "model_trainedAtZeroZ.ckpt")


    def _input_spectrum_info(self, filename, redshift, n):
        loadInputSpectra = LoadInputSpectra(filename, redshift, redshift, self.smooth)
        inputImage, inputRedshift, typeNamesList, nw, nBins = loadInputSpectra.input_spectra()
        bestTypesList = BestTypesListSingleRedshift(self.modelFilename, inputImage, typeNamesList, nw, nBins)
        bestTypes = bestTypesList.bestTypes
        softmax = bestTypesList.softmaxOrdered

        bestMatchList = []
        for i in range(n):
            name, age = bestTypes[i].split(': ')
            prob = softmax[i]
            bestMatchList.append((name, age, prob))

        return np.array(bestMatchList)

    def list_best_matches(self, n=1):
        """Returns a list of lists of the the top n best matches for each spectrum"""
        bestMatchLists = []
        for i in range(self.numSpectra):
            f = self.filenames[i]
            z = self.redshifts[i]
            bestMatchList = self._input_spectrum_info(f, z, n)
            bestMatchLists.append(bestMatchList)

        return np.array(bestMatchLists)

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
# print classification.list_best_matches()




