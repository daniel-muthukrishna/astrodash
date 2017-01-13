import sys
import os
import numpy as np
mainDirectory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(mainDirectory, ".."))

from restore_model import LoadInputSpectra, BestTypesListSingleRedshift


class Classify(object):
    def __init__(self, filenames=[], redshifts=[]):
        """ Takes a list of filenames and corresponding redshifts for supernovae.
        Files should contain a single spectrum, and redshifts should be a list of corresponding redshift floats
        """
        self.filenames = filenames
        self.redshifts = redshifts
        self.numSpectra = len(filenames)
        self.mainDirectory = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(self.mainDirectory, ".."))
        self.modelFilename = os.path.join(self.mainDirectory, "../model_trainedAtZeroZ.ckpt")

    def _input_spectrum_info(self, filename, redshift, n, smooth=10):
        loadInputSpectra = LoadInputSpectra(filename, redshift, redshift, smooth)
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





# # EXAMPLE USAGE:
# classification = Classify(filenames=['/Users/dmuthukrishna/Users/dmuthukrishna/DES16E1dic_E1_combined_161125_v10_b00.dat',
#                                      '/Users/dmuthukrishna/Users/dmuthukrishna/DES16E1dic_E1_combined_161125_v10_b00.dat'],
#                           redshifts=[0.34, 0.13])
# print classification.list_best_matches()




