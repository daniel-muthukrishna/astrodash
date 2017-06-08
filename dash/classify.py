import os
import sys
import pickle
import numpy as np
from dash.download_data_files import download_all_files
from dash.restore_model import LoadInputSpectra, BestTypesListSingleRedshift, get_training_parameters, classification_split
from dash.false_positive_rejection import FalsePositiveRejection, combined_prob
from dash.read_binned_templates import load_templates, get_templates, ReadBinnedTemplates
from dash.calculate_redshift import get_median_redshift

try:
    from PyQt5 import QtGui
    from dash.gui_main import MainApp
except ImportError:
    print("Warning: You will need to install 'PyQt5' if you want to use the graphical interface. "
          "Using the automatic library will continue to work.")


class Classify(object):
    def __init__(self, filenames=[], redshifts=[], smooth=6, minWave=2500, maxWave=10000, classifyHost=False, knownZ=True, data_files='models_v01'):
        """ Takes a list of filenames and corresponding redshifts for supernovae.
        Files should contain a single spectrum, and redshifts should be a list of corresponding redshift floats
        """
        # download_all_files('v01')
        self.filenames = filenames
        self.redshifts = redshifts
        self.smooth = smooth
        self.minWave = minWave
        self.maxWave = maxWave
        self.classifyHost = classifyHost
        self.numSpectra = len(filenames)
        self.scriptDirectory = os.path.dirname(os.path.abspath(__file__))
        if knownZ and redshifts != []:
            self.knownZ = True
        else:
            self.knownZ = False
        self.pars = get_training_parameters()
        self.nw, w0, w1 = self.pars['nw'], self.pars['w0'], self.pars['w1']
        self.dwlog = np.log(w1/w0)/self.nw
        self.snTemplates, self.galTemplates = load_templates(os.path.join(self.scriptDirectory, data_files, 'models/sn_and_host_templates.npz'))

        if self.knownZ:
            if classifyHost:
                self.modelFilename = os.path.join(self.scriptDirectory, data_files, "models/zeroZ_classifyHost/tensorflow_model.ckpt")
            else:
                self.modelFilename = os.path.join(self.scriptDirectory, data_files, "models/zeroZ/tensorflow_model.ckpt")
        else:
            if self.classifyHost:
                self.modelFilename = os.path.join(self.scriptDirectory, data_files, "models/agnosticZ_classifyHost/tensorflow_model.ckpt")
            else:
                self.modelFilename = os.path.join(self.scriptDirectory, data_files, "models/agnosticZ/tensorflow_model.ckpt")

    def _get_images(self, filename, redshift):
        loadInputSpectra = LoadInputSpectra(filename, redshift, redshift, self.smooth, self.pars, self.minWave, self.maxWave, self.classifyHost)
        inputImage, inputRedshift, typeNamesList, nw, nBins = loadInputSpectra.input_spectra()

        return inputImage, typeNamesList, nw, nBins

    def _input_spectra_info(self):
        inputImages = np.empty((0, int(self.nw)), np.float16)
        for i in range(self.numSpectra):
            f = self.filenames[i]
            if self.knownZ:
                z = self.redshifts[i]
            else:
                z = 0
            inputImage, typeNamesList, nw, nBins = self._get_images(f, z)
            inputImages = np.append(inputImages, inputImage, axis=0)
        bestTypesList = BestTypesListSingleRedshift(self.modelFilename, inputImages, typeNamesList, self.nw, nBins)
        bestTypes = bestTypesList.bestTypes
        softmaxes = bestTypesList.softmaxOrdered
        bestLabels = bestTypesList.idx

        return bestTypes, softmaxes, bestLabels, inputImages

    def list_best_matches(self, n=5):
        """Returns a list of lists of the the top n best matches for each spectrum"""
        bestTypes, softmaxes, bestLabels, inputImages = self._input_spectra_info()
        bestMatchLists = []
        bestBroadTypes = []
        rejectionLabels = []
        reliableFlags = []
        redshifts = []
        for specNum in range(self.numSpectra):
            bestMatchList = []
            for i in range(20):
                host, name, age = classification_split(bestTypes[specNum][i])
                if not self.knownZ:
                    redshifts.append(self.calc_redshift(inputImages[i], name, age)[0])
                prob = softmaxes[specNum][i]
                bestMatchList.append((host, name, age, prob))
            bestMatchList = np.array(bestMatchList)
            bestMatchLists.append(bestMatchList[0:n])
            bestBroadType, reliableFlag = self.best_broad_type(bestMatchList)
            bestBroadTypes.append(bestBroadType)
            reliableFlags.append(reliableFlag)
            rejectionLabels.append(self.false_positive_rejection(bestTypes[specNum][0], inputImages[specNum]))

        bestMatchLists = np.array(bestMatchLists)

        if not redshifts:
            redshifts = self.redshifts
        else:
            redshifts = np.array(redshifts)

        return bestMatchLists, redshifts, bestBroadTypes, rejectionLabels, reliableFlags

    def best_broad_type(self, bestMatchList):
        host, prevName, bestAge, probTotal, reliableFlag = combined_prob(bestMatchList[0:10])

        return (prevName, bestAge, probTotal), reliableFlag

    def false_positive_rejection(self, bestType, inputImage):
        host, name, age = classification_split(bestType)
        snInfos, snNames, hostInfos, hostNames = get_templates(name, age, host, self.snTemplates, self.galTemplates, self.nw)
        if snInfos != []:
            templateImages = snInfos[:, 1]
            falsePositiveRejection = FalsePositiveRejection(inputImage, templateImages)
            rejectionLabel = "NONE"  # "(chi2=%s, rlap=%s)" % (falsePositiveRejection.rejection_label(), falsePositiveRejection.rejection_label2())
        else:
            rejectionLabel = "(NO_TEMPLATES)"

        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # plt.plot(inputImage)
        # plt.plot(templateImage)
        # plt.show()

        return rejectionLabel

    def calc_redshift(self, inputFlux, snName, snAge):
        host = "No Host"
        snInfos, snNames, hostInfos, hostNames = get_templates(snName, snAge, host, self.snTemplates, self.galTemplates, self.nw)
        numOfSubTemplates = len(snNames)
        templateFluxes = []
        for i in range(numOfSubTemplates):
            templateFluxes.append(snInfos[i][1])

        redshift, crossCorr = get_median_redshift(inputFlux, templateFluxes, self.nw, self.dwlog)
        print(redshift)
        if redshift is None:
            return 0, np.zeros(1024)

        return round(redshift, 4), crossCorr

    # def save_best_matches(self, n=1, filename='DASH_matches.txt'):
    #     if ('self.bestMatchLists' not in locals()) or self.n != n:
    #         self.bestMatchLists = self.list_best_matches(n)
    #     np.savetxt(filename, self.bestMatchLists)

    def plot_with_gui(self, indexToPlot=0):
        app = QtGui.QApplication(sys.argv)
        form = MainApp(inputFilename=self.filenames[indexToPlot])
        form.lblInputFilename.setText(self.filenames[indexToPlot].split('/')[-1])
        form.checkBoxKnownZ.setChecked(self.knownZ)
        form.checkBoxClassifyHost.setChecked(self.classifyHost)
        form.lineEditKnownZ.setText(str(self.redshifts[indexToPlot]))
        form.lineEditSmooth.setText(str(self.smooth))
        form.classifyHost = self.classifyHost
        form.select_tensorflow_model()
        form.fit_spectra()
        form.show()
        app.exec_()




# # EXAMPLE USAGE:
# classification = Classify(filenames=['/Users/dmuthukrishna/Users/dmuthukrishna/DES16E1dic_E1_combined_161125_v10_b00.dat',
#                                      '/Users/dmuthukrishna/Users/dmuthukrishna/DES16E1dic_E1_combined_161125_v10_b00.dat'],
#                           redshifts=[0.34, 0.13])
# print(classification.list_best_matches())




