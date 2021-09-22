import os
import pickle
from astrodash.input_spectra import *
from tensorflow.keras.models import load_model



def get_training_parameters(data_files='models_v06'):
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(scriptDirectory, data_files, "models/zeroZ/training_params.pickle"), 'rb') as f:
        pars = pickle.load(f)
    return pars


class LoadInputSpectra(object):
    def __init__(self, inputFilename, z, smooth, pars, minWave, maxWave, classifyHost):
        self.nw = pars['nw']
        nTypes, w0, w1, minAge, maxAge, ageBinSize, typeList = pars['nTypes'], pars['w0'], pars['w1'], pars['minAge'], \
                                                               pars['maxAge'], pars['ageBinSize'], pars['typeList']

        if classifyHost:
            hostList = pars['galTypeList']
            nHostTypes = len(hostList)
        else:
            hostList, nHostTypes = None, 1

        self.inputSpectra = InputSpectra(inputFilename, z, nTypes, minAge, maxAge, ageBinSize, w0, w1, self.nw,
                                         typeList, smooth, minWave, maxWave, hostList, nHostTypes)

        self.inputImages, self.inputFilenames, self.inputRedshifts, self.typeNamesList, self.inputMinMaxIndexes = self.inputSpectra.redshifting()
        self.nBins = len(self.typeNamesList)

    def input_spectra(self):
        return self.inputImages, self.inputRedshifts, self.typeNamesList, int(
            self.nw), self.nBins, self.inputMinMaxIndexes


class RestoreModel(object):
    def __init__(self, modelFilename, inputImages, nw, nBins):
        self.modelFilename = modelFilename
        self.inputImages = inputImages
        self.nw = nw
        self.nBins = nBins

        self.model = load_model(modelFilename)

    def restore_variables(self):
        softmax = self.model.predict(self.inputImages)

        return softmax


class BestTypesListSingleRedshift(object):
    def __init__(self, modelFilename, inputImages, typeNamesList, nw, nBins):
        self.modelFilename = modelFilename
        self.inputImages = inputImages
        self.typeNamesList = typeNamesList
        self.nBins = nBins

        self.restoreModel = RestoreModel(self.modelFilename, self.inputImages, nw, nBins)
        self.typeNamesList = np.array(typeNamesList)

        # if more than one image, then variables will be a list of length len(inputImages)
        softmaxes = self.restoreModel.restore_variables()
        self.bestTypes, self.softmaxOrdered, self.idx = [], [], []
        for softmax in softmaxes:
            bestTypes, idx, softmaxOrdered = self.create_list(softmax)
            self.bestTypes.append(bestTypes)
            self.softmaxOrdered.append(softmaxOrdered)
            self.idx.append(idx)

    def create_list(self, softmax):
        idx = np.argsort(softmax)  # list of the index of the highest probabiites
        bestTypes = self.typeNamesList[idx[::-1]]  # reordered in terms of softmax probability columns

        return bestTypes, idx, softmax[idx[::-1]]


def classification_split(classificationString):
    classification = classificationString.split(': ')
    if len(classification) == 2:
        snName, snAge = classification
        host = ""
    else:
        host, snName, snAge = classification

    return host, snName, snAge
