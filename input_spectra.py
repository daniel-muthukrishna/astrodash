import numpy as np
import pickle
from create_arrays import *


class InputSpectra(object):
    def __init__(self, filename, minZ, maxZ, nTypes, minAge, maxAge, ageBinSize, w0, w1, nw):
        self.filename = filename
        self.minZ = minZ
        self.maxZ = maxZ
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize
        self.ageBinning = AgeBinning(self.minAge, self.maxAge, self.ageBinSize)
        self.numOfAgeBins = self.ageBinning.age_bin(self.maxAge) + 1
        self.nLabels = self.nTypes * self.numOfAgeBins
        self.redshiftPrecision = 1000
        self.numOfRedshifts = (self.maxZ - self.minZ) * self.redshiftPrecision
        self.createLabels = CreateLabels(self.nTypes, self.minAge, self.maxAge, self.ageBinSize)
        self.fileType = 'fits or twocolumn etc.' #Will use later on
        self.typeNamesList = self.createLabels.type_names_list()
        

    def redshifting(self):
        images = np.empty((0, int(self.nw)), np.float32)  # Number of pixels
        labels = np.empty((0, self.nLabels), float)  # Number of labels (SN types)
        filenames = []
        typeNames = []
        redshifts = []

        for z in np.linspace(self.minZ, self.maxZ, self.numOfRedshifts + 1):
            readSpectra = ReadSpectra(self.w0, self.w1, self.nw, z)
            tempwave, tempflux, tminindex, tmaxindex, age, snName, ttype = readSpectra.input_spectrum(sfTemplateLocation, self.filename)
            label, typeName = self.createLabels.label_array(ttype, age)
            nonzeroflux = tempflux[tminindex:tmaxindex + 1]
            newflux = (nonzeroflux - min(nonzeroflux)) / (max(nonzeroflux) - min(nonzeroflux))
            newflux2 = np.concatenate((tempflux[0:tminindex], newflux, tempflux[tmaxindex + 1:]))
            images = np.append(images, np.array([newflux2]), axis=0)  # images.append(newflux2)
            labels = np.append(labels, np.array([label]), axis=0)  # labels.append(ttype)
            filenames.append(self.filename + "_" + str(z))
            typeNames.append(typeName)
            redshifts.append(z)

        inputImages = np.array(images)
        inputLabels = np.array(labels)
        inputFilenames = np.array(filenames)
        inputTypeNames = np.array(typeNames)
        inputRedshifts = np.array(redshifts)

        return inputImages, inputLabels, inputFilenames, inputTypeNames, inputRedshifts


    def saveArrays(self):
        inputImages, inputLabels, inputFilenames, inputTypeNames, inputRedshifts = self.redshifting()
        np.savez_compressed('input_data.npz', inputImages=inputImages, inputLabels=inputLabels,
                            inputFilenames=inputFilenames, inputTypeNames=inputTypeNames, inputRedshifts=inputRedshifts,
                            typeNamesList = self.typeNamesList)

sfTemplateLocation = '/home/dan/Desktop/SNClassifying_Pre-alpha/templates/superfit_templates/sne/'
sfFilename = 'Ia/sn1981b.max.dat'

filename = sfFilename

with open('training_params.pickle') as f:
    nTypes, w0, w1, nw, minAge, maxAge, ageBinSize = pickle.load(f)
    
minZ = 0
maxZ = 0.5

InputSpectra(filename, minZ, maxZ, nTypes, minAge, maxAge, ageBinSize, w0, w1, nw).saveArrays()
