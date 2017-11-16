import numpy as np
from random import shuffle
import multiprocessing as mp
import itertools
import time
from dash.sn_processing import PreProcessing
from dash.combine_sn_and_host import training_template_data
from dash.preprocessing import ProcessingTools
from dash.array_tools import zero_non_overlap_part, normalise_spectrum


class AgeBinning(object):
    def __init__(self, minAge, maxAge, ageBinSize):
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize

    def age_bin(self, age):
        ageBin = int(round(age / self.ageBinSize)) - int(round(self.minAge / self.ageBinSize))

        return ageBin

    def age_labels(self):
        ageLabels = []

        ageBinPrev = 0
        ageLabelMin = self.minAge
        for age in np.arange(self.minAge, self.maxAge, 0.5):
            ageBin = self.age_bin(age)

            if ageBin != ageBinPrev:
                ageLabelMax = int(round(age))
                ageLabels.append(str(int(ageLabelMin)) + " to " + str(ageLabelMax))
                ageLabelMin = ageLabelMax

            ageBinPrev = ageBin

        ageLabels.append(str(int(ageLabelMin)) + " to " + str(int(self.maxAge)))

        return ageLabels


class CreateLabels(object):

    def __init__(self, nTypes, minAge, maxAge, ageBinSize, typeList, hostList, nHostTypes):
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize
        self.typeList = typeList
        self.ageBinning = AgeBinning(self.minAge, self.maxAge, self.ageBinSize)
        self.numOfAgeBins = self.ageBinning.age_bin(self.maxAge-0.1) + 1
        self.nLabels = self.nTypes * self.numOfAgeBins
        self.ageLabels = self.ageBinning.age_labels()
        self.hostList = hostList
        self.nHostTypes = nHostTypes

    def label_array(self, ttype, age, host=None):
        ageBin = self.ageBinning.age_bin(age)

        try:
            typeIndex = self.typeList.index(ttype)
        except ValueError as err:
            raise Exception("INVALID TYPE: {0}".format(err))

        if host is None:
            labelArray = np.zeros((self.nTypes, self.numOfAgeBins))
            labelArray[typeIndex][ageBin] = 1
            labelArray = labelArray.flatten()
            typeName = ttype + ": " + self.ageLabels[ageBin]
        else:
            hostIndex = self.hostList.index(host)
            labelArray = np.zeros((self.nHostTypes, self.nTypes, self.numOfAgeBins))
            labelArray[hostIndex][typeIndex][ageBin] = 1
            labelArray = labelArray.flatten()
            typeName = "{}: {}: {}".format(host, ttype, self.ageLabels[ageBin])

        labelIndex = np.argmax(labelArray)

        return labelIndex, typeName

    def type_names_list(self):
        typeNamesList = []
        if self.hostList is None:
            for tType in self.typeList:
                for ageLabel in self.ageBinning.age_labels():
                    typeNamesList.append("{}: {}".format(tType, ageLabel))
        else:
            for host in self.hostList:
                for tType in self.typeList:
                    for ageLabel in self.ageBinning.age_labels():
                        typeNamesList.append("{}: {}: {}".format(host, tType, ageLabel))

        return np.array(typeNamesList)
        

class TempList(object):
    def temp_list(self, tempFileList):
        f = open(tempFileList, 'rU')

        fileList = f.readlines()
        for i in range(0,len(fileList)):
            fileList[i] = fileList[i].strip('\n')

        f.close()

        return fileList
    

class ReadSpectra(object):

    def __init__(self, w0, w1, nw, snFilename, galFilename=None):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.snFilename = snFilename
        if galFilename is None:
            self.data = PreProcessing(snFilename, w0, w1, nw)
        else:
            self.galFilename = galFilename

    def sn_plus_gal_template(self, snAgeIdx, snCoeff, galCoeff, z):
        wave, flux, minIndex, maxIndex, nCols, ages, tType = training_template_data(snAgeIdx, snCoeff, galCoeff, z, self.snFilename, self.galFilename, self.w0, self.w1, self.nw)

        return wave, flux, nCols, ages, tType, minIndex, maxIndex

    def input_spectrum(self, z, smooth, minWave, maxWave):
        wave, flux, minIndex, maxIndex = self.data.two_column_data(z, smooth, minWave, maxWave)

        return wave, flux, int(minIndex), int(maxIndex)


class ArrayTools(object):

    def __init__(self, nLabels, nw):
        self.nLabels = nLabels
        self.nw = nw

    def shuffle_arrays(self, **kwargs):
        """ Must take images and labels as arguments with the keyword specified.
        Can optionally take filenames and typeNames as arguments """
        arraySize = len(kwargs['labels'])
        kwargShuf = {}
        for key in kwargs:
            if key == 'images':
                arrayShuf = np.zeros((arraySize, int(self.nw)), np.float16)
            elif key == 'labels':
                arrayShuf = np.zeros(arraySize, np.uint16)
            else:
                arrayShuf = np.empty(arraySize, dtype=object)
            kwargShuf[key] = arrayShuf
        idx = 0
        # Randomise order
        indexShuf = list(range(arraySize))
        shuffle(indexShuf)
        for i in indexShuf:
            for key in kwargs:
                kwargShuf[key][idx] = kwargs[key][i]
            idx += 1

        return kwargShuf

    def count_labels(self, labels):
        counts = np.zeros(self.nLabels)

        for i in range(len(labels)):
            counts[labels[i]] += 1

        return counts

    def div0(self, a, b):
        """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[~ np.isfinite(c)] = 0  # -inf inf NaN
        return c

    def augment_data(self, flux, stdDevMean=0.05, stdDevStdDev=0.05):
        minIndex, maxIndex = ProcessingTools().min_max_index(flux, outerVal=0.5)
        noise = np.zeros(self.nw)
        stdDev = abs(np.random.normal(stdDevMean, stdDevStdDev)) # randomised standard deviation
        noise[minIndex:maxIndex] = np.random.normal(0, stdDev, maxIndex - minIndex)
        # # Add white noise to regions outside minIndex to maxIndex
        # noise[0:minIndex] = np.random.uniform(0.0, 1.0, minIndex)
        # noise[maxIndex:] = np.random.uniform(0.0, 1.0, self.nw-maxIndex)

        augmentedFlux = flux + noise
        augmentedFlux = normalise_spectrum(augmentedFlux)
        augmentedFlux = zero_non_overlap_part(augmentedFlux, minIndex, maxIndex, outerVal=0.5)

        return augmentedFlux

    def over_sample_arrays(self, **kwargs):
        """ Must take images and labels as arguments with the keyword specified.
        Can optionally take filenames and typeNames as arguments """
        counts = self.count_labels(kwargs['labels'])
        idx = 0
        print("Before OverSample")  #
        print(counts)  #

        overSampleAmount = self.div0(1 * max(counts), counts)  # ignore zeros in counts
        overSampleArraySize = int(sum(np.array(overSampleAmount, int) * counts))
        print(np.array(overSampleAmount, int) * counts)
        print(np.array(overSampleAmount, int))
        print(overSampleArraySize, len(kwargs['labels']))
        kwargOverSampled = {}
        for key in kwargs:
            if key == 'images':
                arrayOverSampled = np.zeros((overSampleArraySize, int(self.nw)), np.float16)
            elif key == 'labels':
                arrayOverSampled = np.zeros(overSampleArraySize, np.uint16)
            else:
                arrayOverSampled = np.empty(overSampleArraySize, dtype=object)
            kwargOverSampled[key] = arrayOverSampled

        counts1 = np.zeros(self.nLabels)

        kwargShuf = self.shuffle_arrays(**kwargs)

        print(len(kwargShuf['labels']))
        for i in range(len(kwargShuf['labels'])):
            labelIndex = kwargShuf['labels'][i]

            if overSampleAmount[labelIndex] < 10:
                std = 0.03
            else:
                std = 0.05
            for r in range(int(overSampleAmount[labelIndex])):
                for key in kwargs:
                    if key == 'images':
                        kwargOverSampled[key][idx] = self.augment_data(kwargShuf[key][i], stdDevMean=0.05, stdDevStdDev=std)
                    else:
                        kwargOverSampled[key][idx] = kwargShuf[key][i]
                counts1[labelIndex] += 1
                idx += 1

        print("After OverSample")  #
        print(counts1)  #

        print("Before Shuffling")
        kwargOverSampledShuf = self.shuffle_arrays(**kwargOverSampled)
        print("After Shuffling")
        return kwargOverSampledShuf


class CreateArrays(object):
    def __init__(self, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, redshiftPrecision, hostTypes=None, nHostTypes=None):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize
        self.typeList = typeList
        self.minZ = minZ
        self.maxZ = maxZ
        self.numOfRedshifts = int((maxZ - minZ) * 1./redshiftPrecision)
        self.ageBinning = AgeBinning(minAge, maxAge, ageBinSize)
        self.numOfAgeBins = self.ageBinning.age_bin(maxAge-0.1) + 1
        self.nLabels = nTypes * self.numOfAgeBins * nHostTypes
        self.createLabels = CreateLabels(self.nTypes, self.minAge, self.maxAge, self.ageBinSize, self.typeList, hostTypes, nHostTypes)
        self.hostTypes = hostTypes

    def combined_sn_gal_templates_to_arrays(self, snTemplateLocation, snTempFileList, galTemplateLocation, galTempList, snFractions):
        snTempList = TempList().temp_list(snTempFileList)
        typeList = []
        images = np.empty((0, int(self.nw)), np.float16)  # Number of pixels
        labelsIndexes = [] # labels = np.empty((0, self.nLabels), np.uint8)  # Number of labels (SN types)
        filenames = []  # np.empty(0)
        typeNames = []  # np.empty(0)
        agesList = []

        for j in range(len(galTempList)):
            galFilename = galTemplateLocation + galTempList[j] if galTemplateLocation is not None else None
            for i in range(0, len(snTempList)):
                nCols = 15
                readSpectra = ReadSpectra(self.w0, self.w1, self.nw, snTemplateLocation + snTempList[i], galFilename)
                for ageidx in range(0, 1000):
                    if ageidx < nCols:
                        for snCoeff in snFractions:
                            galCoeff = 1 - snCoeff
                            for z in np.linspace(self.minZ, self.maxZ, self.numOfRedshifts + 1):
                                tempWave, tempFlux, nCols, ages, tType, tMinIndex, tMaxIndex = readSpectra.sn_plus_gal_template(ageidx, snCoeff, galCoeff, z)
                                agesList.append(ages[ageidx])
                                if tMinIndex == tMaxIndex or not tempFlux.any():
                                    print("NO DATA for {} {} ageIdx:{} z>={}".format(galTempList[j], snTempList[i], ageidx, z))
                                    break

                                if self.minAge < float(ages[ageidx]) < self.maxAge:
                                    if self.hostTypes is None:  # Checks if we are classifying by host as well
                                        labelIndex, typeName = self.createLabels.label_array(tType, ages[ageidx], host=None)
                                    else:
                                        labelIndex, typeName = self.createLabels.label_array(tType, ages[ageidx], host=galTempList[j])
                                    if tMinIndex > (self.nw - 1):
                                        continue
                                    nonzeroflux = tempFlux[tMinIndex:tMaxIndex + 1]
                                    newflux = (nonzeroflux - min(nonzeroflux)) / (max(nonzeroflux) - min(nonzeroflux))
                                    newflux2 = np.concatenate((tempFlux[0:tMinIndex], newflux, tempFlux[tMaxIndex + 1:]))
                                    images = np.append(images, np.array([newflux2]), axis=0)
                                    labelsIndexes.append(labelIndex) # labels = np.append(labels, np.array([label]), axis=0)
                                    filenames.append("{0}_{1}_{2}_{3}_snCoeff{4}_z{5}".format(snTempList[i], tType, str(ages[ageidx]), galTempList[j], snCoeff, (z)))
                                    typeNames.append(typeName)

                        print(snTempList[i], ageidx, nCols, galTempList[j], snCoeff)
                    else:
                        break

        return typeList, images, np.array(labelsIndexes), np.array(filenames), np.array(typeNames)

    def combined_sn_gal_arrays_multiprocessing(self, snTemplateLocation, snTempFileList, galTemplateLocation, galTempFileList):
        if galTemplateLocation is None or galTempFileList is None:
            return self.combined_sn_gal_templates_to_arrays(snTemplateLocation, snTempFileList, galTemplateLocation=None, galTempList=[None], snFractions=[1.0])

        galTempList = TempList().temp_list(galTempFileList)
        snFractions = [0.99, 0.98, 0.95, 0.93, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        galAndSnFracParams = list(itertools.product(galTempList, snFractions))

        images = np.empty((0, int(self.nw)), np.float16)
        labelsIndexes = np.empty(0, np.uint16)
        filenames = np.empty(0)
        typeNames = np.empty(0)

        t1 = time.time()
        pool = mp.Pool()
        results = [pool.apply_async(self.combined_sn_gal_templates_to_arrays, args=(snTemplateLocation, snTempFileList, galTemplateLocation, [gal], [snFrac])) for gal, snFrac in galAndSnFracParams]
        pool.close()
        pool.join()

        outputs = [p.get() for p in results]

        for out in outputs:
            typeList, imagesPart, labelsPart, filenamesPart, typeNamesPart = out
            images = np.append(images, imagesPart, axis=0)
            labelsIndexes = np.append(labelsIndexes, labelsPart, axis=0)
            filenames = np.append(filenames, filenamesPart)
            typeNames = np.append(typeNames, typeNamesPart)

        t2 = time.time()
        print("time spent: {0:.2f}".format(t2 - t1))

        try:
            print("SIZE OF ARRAYS:")
            print(images.nbytes)
            print(labelsIndexes.nbytes)
            print(filenames.nbytes)
            print(typeNames.nbytes)
        except Exception as e:
            print("Exception Raised: {0}".format(e))
        print("Completed Creating Arrays!")

        return typeList, images, labelsIndexes, filenames, typeNames
