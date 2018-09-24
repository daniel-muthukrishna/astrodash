import os
import sys
import pickle
import numpy as np
from astrodash.download_data_files import download_all_files
from astrodash.restore_model import LoadInputSpectra, BestTypesListSingleRedshift, get_training_parameters, classification_split
from astrodash.false_positive_rejection import RlapCalc, combined_prob
from astrodash.read_binned_templates import load_templates, get_templates
from astrodash.calculate_redshift import get_median_redshift
from astrodash.read_from_catalog import catalogDict

try:
    from PyQt5 import QtGui
    from astrodash.gui_main import MainApp
except ImportError:
    print("Warning: You will need to install 'PyQt5' if you want to use the graphical interface. "
          "Using the automatic library will continue to work.")


class Classify(object):
    def __init__(self, filenames=[], redshifts=[], smooth=6, minWave=3500, maxWave=10000, classifyHost=False, knownZ=True, rlapScores=True, data_files='models_v06'):
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
        self.rlapScores = rlapScores
        self.pars = get_training_parameters()
        self.nw, w0, w1 = self.pars['nw'], self.pars['w0'], self.pars['w1']
        self.dwlog = np.log(w1/w0)/self.nw
        self.wave = w0 * np.exp(np.arange(0, self.nw) * self.dwlog)
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
        if redshift in list(catalogDict.keys()):
            redshift = 0
        loadInputSpectra = LoadInputSpectra(filename, redshift, self.smooth, self.pars, self.minWave, self.maxWave, self.classifyHost)
        inputImage, inputRedshift, typeNamesList, nw, nBins, inputMinMaxIndex = loadInputSpectra.input_spectra()

        return inputImage, typeNamesList, nw, nBins, inputMinMaxIndex, inputRedshift

    def _input_spectra_info(self):
        inputImages = np.empty((0, int(self.nw)), np.float16)
        inputMinMaxIndexes = []
        for i in range(self.numSpectra):
            f = self.filenames[i]
            if self.knownZ:
                z = self.redshifts[i]
            else:
                z = 0
            inputImage, typeNamesList, nw, nBins, inputMinMaxIndex, inputRedshift = self._get_images(f, z)
            self.redshifts[i] = -inputRedshift[0]
            inputImages = np.append(inputImages, inputImage, axis=0)
            inputMinMaxIndexes.append(inputMinMaxIndex[0])
        bestTypesList = BestTypesListSingleRedshift(self.modelFilename, inputImages, typeNamesList, self.nw, nBins)
        bestTypes = bestTypesList.bestTypes
        softmaxes = bestTypesList.softmaxOrdered
        bestLabels = bestTypesList.idx

        return bestTypes, softmaxes, bestLabels, inputImages, inputMinMaxIndexes

    def list_best_matches(self, n=5, saveFilename='DASH_matches.txt'):
        """Returns a list of lists of the the top n best matches for each spectrum"""
        bestTypes, softmaxes, bestLabels, inputImages, inputMinMaxIndexes = self._input_spectra_info()
        bestMatchLists = []
        bestBroadTypes = []
        rlapLabels = []
        matchesReliableLabels = []
        redshifts = []
        redshiftErrs = []
        for specNum in range(self.numSpectra):
            bestMatchList = []
            for i in range(20):
                host, name, age = classification_split(bestTypes[specNum][i])
                if not self.knownZ:
                    redshift, _, redshiftErr = self.calc_redshift(inputImages[i], name, age, inputMinMaxIndexes[i])[0]
                    redshifts.append(redshift)
                    redshiftErrs.append(redshiftErr)
                prob = softmaxes[specNum][i]
                bestMatchList.append((host, name, age, prob))
            bestMatchList = np.array(bestMatchList)
            bestMatchLists.append(bestMatchList[0:n])
            bestBroadType, matchesReliableFlag = self.best_broad_type(bestMatchList)
            bestBroadTypes.append(bestBroadType)
            rlapLabel, rlapWarningBool = self.rlap_warning_label(bestTypes[specNum][0], inputImages[specNum], inputMinMaxIndexes[specNum])

            rlapLabels.append(rlapLabel)
            if matchesReliableFlag:
                matchesReliableLabels.append("Reliable matches")
            else:
                matchesReliableLabels.append("Unreliable matches")

        bestMatchLists = np.array(bestMatchLists)

        if not redshifts:
            redshifts = self.redshifts
        else:
            redshifts = np.array(redshifts)

        if saveFilename:
            self.save_best_matches(bestMatchLists, redshifts, bestBroadTypes, rlapLabels, matchesReliableLabels, saveFilename)

        return bestMatchLists, redshifts, bestBroadTypes, rlapLabels, matchesReliableLabels, redshiftErrs

    def best_broad_type(self, bestMatchList):
        host, prevName, bestAge, probTotal, reliableFlag = combined_prob(bestMatchList[0:10])

        return (prevName, bestAge, probTotal), reliableFlag

    def rlap_warning_label(self, bestType, inputImage, inputMinMaxIndex):
        host, name, age = classification_split(bestType)
        snInfos, snNames, hostInfos, hostNames = get_templates(name, age, host, self.snTemplates, self.galTemplates, self.nw)
        if snInfos != []:
            if self.rlapScores:
                templateImages = snInfos[:, 1]
                templateMinMaxIndexes = list(zip(snInfos[:, 2], snInfos[:, 3]))
                rlapCalc = RlapCalc(inputImage, templateImages, snNames, self.wave, inputMinMaxIndex, templateMinMaxIndexes)
                rlap, rlapWarningBool = rlapCalc.rlap_label()
                if rlapWarningBool:
                    rlapLabel = "Low rlap: {0}".format(rlap)
                else:
                    rlapLabel = "Good rlap: {0}".format(rlap)
            else:
                rlapLabel = "No rlap"
                rlapWarningBool = "None"
        else:
            rlapLabel = "(NO_TEMPLATES)"
            rlapWarningBool = "None"

        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # plt.plot(inputImage)
        # plt.plot(templateImage)
        # plt.show()

        return rlapLabel, rlapWarningBool

    def calc_redshift(self, inputFlux, snName, snAge, inputMinMaxIndex):
        host = "No Host"
        snInfos, snNames, hostInfos, hostNames = get_templates(snName, snAge, host, self.snTemplates, self.galTemplates, self.nw)
        numOfSubTemplates = len(snNames)
        templateNames = snNames
        templateFluxes = []
        templateMinMaxIndexes = []
        for i in range(numOfSubTemplates):
            templateFluxes.append(snInfos[i][1])
            templateMinMaxIndexes.append((snInfos[i][2], snInfos[i][3]))

        redshift, crossCorr, medianName, redshiftErr = get_median_redshift(inputFlux, templateFluxes, self.nw, self.dwlog, inputMinMaxIndex, templateMinMaxIndexes, templateNames, outerVal=0.5)
        print(redshift)
        if redshift is None:
            return 0, np.zeros(1024)

        return round(redshift, 4), crossCorr, round(redshiftErr, 4)

    def save_best_matches(self, bestFits, redshifts, bestTypes, rlapLabels, matchesReliableLabels, saveFilename='DASH_matches.txt'):
        with open(saveFilename, 'w') as f:
            for i in range(len(self.filenames)):
                f.write("%s   z=%s     %s      %s     %s\n %s\n\n" % (
                    str(self.filenames[i]).split('/')[-1], redshifts[i], bestTypes[i], rlapLabels[i], matchesReliableLabels[i], bestFits[i]))
        print("Finished classifying %d spectra!" % len(self.filenames))

    def plot_with_gui(self, indexToPlot=0):
        app = QtGui.QApplication(sys.argv)
        form = MainApp(inputFilename=self.filenames[indexToPlot])
        if not isinstance(self.filenames[indexToPlot], (list, np.ndarray)) and not hasattr(self.filenames[indexToPlot], 'read'):  # Not an array and not a file-handle
            form.lineEditInputFilename.setText(self.filenames[indexToPlot])
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




