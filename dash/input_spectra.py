import numpy as np
from dash.create_arrays import *


class InputSpectra(object):
    def __init__(self, filename, minZ, maxZ, nTypes, minAge, maxAge, ageBinSize, w0, w1, nw, typeList, smooth, minWave, maxWave, hostList, nHostTypes):
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
        self.typeList = typeList
        self.ageBinning = AgeBinning(self.minAge, self.maxAge, self.ageBinSize)
        self.numOfAgeBins = self.ageBinning.age_bin(self.maxAge) + 1
        self.nLabels = self.nTypes * self.numOfAgeBins * nHostTypes
        self.redshiftPrecision = 1000
        self.numOfRedshifts = (self.maxZ - self.minZ) * self.redshiftPrecision
        self.createLabels = CreateLabels(self.nTypes, self.minAge, self.maxAge, self.ageBinSize, self.typeList, hostList, nHostTypes)
        self.fileType = 'fits or twocolumn etc.' #Will use later on
        self.typeNamesList = self.createLabels.type_names_list()
        self.smooth = smooth
        self.minWave = minWave
        self.maxWave = maxWave

    def redshifting(self):
        images = np.empty((0, int(self.nw)), np.float16)  # Number of pixels
        labels = np.empty((0, self.nLabels), np.uint16)  # Number of labels (SN types)
        filenames = []
        typeNames = []
        redshifts = []
        minMaxIndexes = []
        readSpectra = ReadSpectra(self.w0, self.w1, self.nw, self.filename)

        #Undo it's previous redshift)
        for z in np.linspace(self.minZ, self.maxZ, self.numOfRedshifts + 1):
            wave, flux, minIndex, maxIndex = readSpectra.input_spectrum(z, self.smooth, self.minWave, self.maxWave)
            nonzeroflux = flux[minIndex:maxIndex + 1]
            newflux = (nonzeroflux - min(nonzeroflux)) / (max(nonzeroflux) - min(nonzeroflux))
            newflux2 = np.concatenate((flux[0:minIndex], newflux, flux[maxIndex + 1:]))
            images = np.append(images, np.array([newflux2]), axis=0)  # images.append(newflux2)
            filenames.append(self.filename + "_" + str(-z))
            redshifts.append(-z)
            minMaxIndexes.append((minIndex, maxIndex))

        inputImages = np.array(images)
        inputFilenames = np.array(filenames)
        inputRedshifts = np.array(redshifts)

        return inputImages, inputFilenames, inputRedshifts, self.typeNamesList, minMaxIndexes


    def saveArrays(self):
        inputImages, inputFilenames, inputRedshifts, minMaxIndex = self.redshifting()
        np.savez_compressed('input_data.npz', inputImages=inputImages, inputFilenames=inputFilenames,
                            inputRedshifts=inputRedshifts, typeNamesList = self.typeNamesList)
##
#sfTemplateLocation = '/home/dan/Desktop/SNClassifying_Pre-alpha/templates/superfit_templates/sne/'
##sfFilename = 'Ia/sn1981b.max.dat'
##
##filename = sfFilename
##
##with open('data_files/training_params.pickle') as f:
##    nTypes, w0, w1, nw, minAge, maxAge, ageBinSize = pickle.load(f)
##    
##minZ = 0
##maxZ = 0.5
##
##InputSpectra(filename, minZ, maxZ, nTypes, minAge, maxAge, ageBinSize, w0, w1, nw).saveArrays()
