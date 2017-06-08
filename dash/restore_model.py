import os
import pickle
import tensorflow as tf
from dash.input_spectra import *
from dash.multilayer_convnet import convnet_variables


def get_training_parameters(data_files='models_v01'):
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(scriptDirectory, data_files, "models/zeroZ/training_params.pickle"), 'rb') as f:
        pars = pickle.load(f)
    return pars


class LoadInputSpectra(object):
    def __init__(self, inputFilename, minZ, maxZ, smooth, pars, minWave, maxWave, classifyHost):
        self.nw = pars['nw']
        nTypes, w0, w1, minAge, maxAge, ageBinSize, typeList = pars['nTypes'], pars['w0'], pars['w1'], pars['minAge'], \
                                                               pars['maxAge'], pars['ageBinSize'], pars['typeList']

        if classifyHost:
            hostList = pars['galTypeList']
            nHostTypes = len(hostList)
        else:
            hostList, nHostTypes = None, 1

        self.inputSpectra = InputSpectra(inputFilename, minZ, maxZ, nTypes, minAge, maxAge, ageBinSize, w0, w1, self.nw, typeList, smooth, minWave, maxWave, hostList, nHostTypes)

        self.inputImages, self.inputFilenames, self.inputRedshifts, self.typeNamesList = self.inputSpectra.redshifting()
        self.nBins = len(self.typeNamesList)

    def input_spectra(self):
        return self.inputImages, self.inputRedshifts, self.typeNamesList, int(self.nw), self.nBins


class RestoreModel(object):
    def __init__(self, modelFilename, inputImages, nw, nBins):
        self.reset()

        self.modelFilename = modelFilename
        self.inputImages = inputImages
        self.nw = nw
        self.nBins = nBins
        self.imWidthReduc = 8
        self.imWidth = 32 #Image size and width

        self.x, self.y_, self.keep_prob, self.y_conv = convnet_variables(self.imWidth, self.imWidthReduc, self.nw, self.nBins)

        self.saver = tf.train.Saver()

    def restore_variables(self):
        with tf.Session() as sess:
            self.saver.restore(sess, self.modelFilename)

            softmax = self.y_conv.eval(feed_dict={self.x: self.inputImages, self.keep_prob: 1.0})
            # print(softmax)
            
        return softmax

    def reset(self):
        tf.reset_default_graph()


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
        idx = np.argsort(softmax) #list of the index of the highest probabiites
        bestTypes = self.typeNamesList[idx[::-1]] #reordered in terms of softmax probability columns

        return bestTypes, idx, softmax[idx[::-1]]


def classification_split(classificationString):
    classification = classificationString.split(': ')
    if len(classification) == 2:
        snName, snAge = classification
        host = ""
    else:
        host, snName, snAge = classification

    return host, snName, snAge