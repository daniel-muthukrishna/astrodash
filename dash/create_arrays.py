import numpy as np
from random import shuffle
from dash.sn_processing import PreProcessing
from dash.combine_sn_and_host import CombineSnAndHost
from dash.preprocessing import ProcessingTools
from dash.array_tools import zero_non_overlap_part, normalise_spectrum
import multiprocessing as mp
import random
import time


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
            print("INVALID TYPE: {0}".format(err))
            raise ValueError

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

        return typeNamesList
        

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
            self.dataCombined = CombineSnAndHost(snFilename, galFilename, w0, w1, nw)

    def sn_plus_gal_template(self, snAgeIdx, snCoeff, galCoeff, z):
        wave, flux, minIndex, maxIndex, nCols, ages, tType = self.dataCombined.training_template_data(snAgeIdx, snCoeff, galCoeff, z)

        return wave, flux, nCols, ages, tType, minIndex, maxIndex

    def snid_template_data(self, ageIdx, z):
        """ lnw template files """
        wave, flux, nCols, ages, tType, minIndex, maxIndex = self.data.snid_template_data(ageIdx, z)

        return wave, flux, nCols, ages, tType, minIndex, maxIndex

    def sf_age(self):
        snName, extension = self.snFilename.strip('.dat').split('.')
        ttype, snName = snName.split('/')

        if extension == 'max':
            age = 0
        elif extension[0] == 'p':
            age = float(extension[1:])
        elif extension[0] == 'm':
            age = -float(extension[1:])
        else:
            print("Invalid Superfit Filename: " + self.snFilename)

        return snName, ttype, age

    def superfit_template_data(self, z):
        """ Returns wavelength and flux after all preprocessing """
        wave, flux, minIndex, maxIndex = self.data.two_column_data(z, minWave=self.w0, maxWave=self.nw)
        snName, ttype, age = self.sf_age()

        print(snName, ttype, age)

        return wave, flux, minIndex, maxIndex, age, snName, ttype

    def input_spectrum(self, z, smooth, minWave, maxWave):
        wave, flux, minIndex, maxIndex = self.data.two_column_data(z, smooth, minWave, maxWave)

        return wave, flux, int(minIndex), int(maxIndex)


class ArrayTools(object):

    def __init__(self, nLabels, nw):
        self.nLabels = nLabels
        self.nw = nw

    def shuffle_arrays(self, images, labels, filenames, typeNames):
        arraySize = len(labels)
        imagesShuf = np.empty((arraySize, int(self.nw)), np.float16)
        labelsShuf = np.empty(arraySize, np.uint16)
        filenamesShuf = np.empty(arraySize, dtype=object)
        typeNamesShuf = np.empty(arraySize, dtype=object)
        idx = 0
        print("Shuffle2")
        # Randomise order
        indexShuf = list(range(arraySize))
        shuffle(indexShuf)
        for i in indexShuf:
            imagesShuf[idx] = images[i]
            labelsShuf[idx] = labels[i]
            filenamesShuf[idx] = filenames[i]
            typeNamesShuf[idx] = typeNames[i]
            idx += 1
        print("LenLabels")
        print(len(labels), idx)
            
        print(imagesShuf)
        print(typeNamesShuf)
        print("Shuffle3")    
        return imagesShuf, labelsShuf, filenamesShuf, typeNamesShuf

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
        minIndex, maxIndex = ProcessingTools().min_max_index(flux)
        noise = np.zeros(self.nw)
        stdDev = abs(np.random.normal(stdDevMean, stdDevStdDev)) # randomised standard deviation
        noise[minIndex:maxIndex] = np.random.normal(0, stdDev, maxIndex - minIndex)
        augmentedFlux = flux + noise
        augmentedFlux = normalise_spectrum(augmentedFlux)
        augmentedFlux = zero_non_overlap_part(augmentedFlux, minIndex, maxIndex)

        return augmentedFlux

    def over_sample_arrays(self, images, labels, filenames, typeNames):
        counts = self.count_labels(labels)
        idx = 0
        print("Before OverSample")  #
        print(counts)  #

        overSampleAmount = self.div0(1 * max(counts), counts)  # ignore zeros in counts
        overSampleArraySize = int(sum(np.array(overSampleAmount, int) * counts))
        print(np.array(overSampleAmount, int) * counts)
        print(np.array(overSampleAmount, int))
        print(overSampleArraySize, len(labels))
        imagesOverSampled = np.zeros((overSampleArraySize, int(self.nw)), np.float16)
        labelsOverSampled = np.zeros(overSampleArraySize, np.uint16)
        filenamesOverSampled = np.empty(overSampleArraySize, dtype=object)
        typeNamesOverSampled = np.empty(overSampleArraySize, dtype=object)

        counts1 = np.zeros(self.nLabels)

        imagesShuf, labelsShuf, filenamesShuf, typeNamesShuf = self.shuffle_arrays(images, labels, filenames, typeNames)
        print(len(labelsShuf))
        for i in range(len(labelsShuf)):
            label = labelsShuf[i]
            image = imagesShuf[i]
            filename = filenamesShuf[i]
            typeName = typeNamesShuf[i]

            labelIndex = label # np.argmax(label)
            
            print(idx, i, int(overSampleAmount[labelIndex]))
            if overSampleAmount[labelIndex] < 10:
                std = 0.03
            else:
                std = 0.05
            for r in range(int(overSampleAmount[labelIndex])):
                imagesOverSampled[idx] = self.augment_data(image, stdDevMean=0.05, stdDevStdDev=std)  # image
                labelsOverSampled[idx] = label
                filenamesOverSampled[idx] = filename
                typeNamesOverSampled[idx] = typeName
                counts1[labelIndex] += 1
                idx += 1

        print("After OverSample")  #
        print(counts1)  #

        print("Before Shuffling")
        imagesOverSampledShuf, labelsOverSampledShuf, filenamesOverSampledShuf, typeNamesOverSampledShuf = self.shuffle_arrays(imagesOverSampled, labelsOverSampled, filenamesOverSampled, typeNamesOverSampled)
        print("After Shuffling")
        return imagesOverSampledShuf, labelsOverSampledShuf, filenamesOverSampledShuf, typeNamesOverSampledShuf


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
        self.numOfRedshifts = (maxZ - minZ) * 1./redshiftPrecision
        self.ageBinning = AgeBinning(minAge, maxAge, ageBinSize)
        self.numOfAgeBins = self.ageBinning.age_bin(maxAge-0.1) + 1
        self.nLabels = nTypes * self.numOfAgeBins * nHostTypes
        self.createLabels = CreateLabels(self.nTypes, self.minAge, self.maxAge, self.ageBinSize, self.typeList, hostTypes, nHostTypes)
        self.hostTypes = hostTypes

    def snid_templates_to_arrays(self, snidTemplateLocation, tempfilelist):
        """ This function is for the SNID processed files, which
            have been preprocessed to negatives, and so cannot be
            imaged yet """

        tempList = TempList().temp_list(tempfilelist) #Arbrirary redshift to read filelist
        typeList = []
        images = np.empty((0, int(self.nw)), np.float16)  # Number of pixels
        labelsIndexes = [] # labels = np.empty((0, self.nLabels), np.uint8)  # Number of labels (SN types)
        filenames = []#np.empty(0)
        typeNames = []#np.empty(0)
        agesList = []

        for i in range(0, len(tempList)):
            ncols = 15
            readSpectra = ReadSpectra(self.w0, self.w1, self.nw, snidTemplateLocation + tempList[i])
            
            for ageidx in range(0, 1000):
                if ageidx < ncols:
                    for z in np.linspace(self.minZ, self.maxZ, self.numOfRedshifts + 1):
                        tempwave, tempflux, ncols, ages, ttype, tminindex, tmaxindex = readSpectra.snid_template_data(ageidx, z)
                        agesList.append(ages[ageidx])
                        if not tempflux.any():
                            print("NO DATA for {} ageIdx:{} z>={}".format(tempList[i], ageidx, z))
                            break

                        if self.minAge < float(ages[ageidx]) < self.maxAge:
                            labelIndex, typeName = self.createLabels.label_array(ttype, ages[ageidx])
                            nonzeroflux = tempflux[tminindex:tmaxindex + 1]
                            newflux = (nonzeroflux - min(nonzeroflux)) / (max(nonzeroflux) - min(nonzeroflux))
                            newflux2 = np.concatenate((tempflux[0:tminindex], newflux, tempflux[tmaxindex + 1:]))
                            images = np.append(images, np.array([newflux2]), axis=0)  # images.append(newflux2)
                            labelsIndexes.append(labelIndex) # labels = np.append(labels, np.array([label]), axis=0)  # labels.append(ttype)
                            filenames.append(tempList[i] + '_' + ttype + '_' + str(ages[ageidx]) + '_z' + str(z))
                            typeNames.append(typeName)

                    print(tempList[i], ageidx, ncols)
                else:
                    break

            # Create List of all SN types
            if ttype not in typeList:
                typeList.append(ttype)
        print(len(images))

        try:
            print("SIZE OF ARRAYS:")
            print(images.nbytes)
            print(np.array(labelsIndexes).nbytes)
            print(np.array(filenames).nbytes)
            print(np.array(typeNames).nbytes)
        except:
            print("Exception Raised")

        return typeList, images, np.array(labelsIndexes), np.array(filenames), np.array(typeNames)

    def combined_sn_gal_templates_to_arrays(self, snTemplateLocation, snTempFileList, galTemplateLocation, galTempList):
        snTempList = TempList().temp_list(snTempFileList)
        typeList = []
        images = np.empty((0, int(self.nw)), np.float16)  # Number of pixels
        labelsIndexes = [] # labels = np.empty((0, self.nLabels), np.uint8)  # Number of labels (SN types)
        filenames = []  # np.empty(0)
        typeNames = []  # np.empty(0)
        agesList = []
        for j in range(len(galTempList)):
            for i in range(0, len(snTempList)):
                ncols = 15
                readSpectra = ReadSpectra(self.w0, self.w1, self.nw, snTemplateLocation + snTempList[i], galTemplateLocation + galTempList[j])
                for ageidx in range(0, 1000):
                    if ageidx < ncols:
                        for snCoeff in [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                            galCoeff = 1 - snCoeff
                            for z in np.linspace(self.minZ, self.maxZ, self.numOfRedshifts + 1):
                                tempwave, tempflux, ncols, ages, ttype, tminindex, tmaxindex = readSpectra.sn_plus_gal_template(ageidx, snCoeff, galCoeff, z)
                                agesList.append(ages[ageidx])
                                if not tempflux.any():
                                    print("NO DATA for {} {} ageIdx:{} z>={}".format(galTempList[j], snTempList[i], ageidx, z))
                                    break

                                if self.minAge < float(ages[ageidx]) < self.maxAge:
                                    if self.hostTypes is None: # Checks if we are classifying by host as well
                                        labelIndex, typeName = self.createLabels.label_array(ttype, ages[ageidx], host=None)
                                    else:
                                        labelIndex, typeName = self.createLabels.label_array(ttype, ages[ageidx], host=galTempList[j])
                                    nonzeroflux = tempflux[tminindex:tmaxindex + 1]
                                    newflux = (nonzeroflux - min(nonzeroflux)) / (max(nonzeroflux) - min(nonzeroflux))
                                    newflux2 = np.concatenate((tempflux[0:tminindex], newflux, tempflux[tmaxindex + 1:]))
                                    images = np.append(images, np.array([newflux2]), axis=0)
                                    labelsIndexes.append(labelIndex) # labels = np.append(labels, np.array([label]), axis=0)
                                    filenames.append("{0}_{1}_{2}_{3}_snCoeff{4}_z{5}".format(snTempList[i], ttype, str(ages[ageidx]), galTempList[j], snCoeff, (z)))
                                    typeNames.append(typeName)

                        print(snTempList[i], ageidx, ncols, galTempList[j], snCoeff)
                    else:
                        break

                # Create List of all SN types
                if ttype not in typeList:
                    typeList.append(ttype)
        print(len(images))

        try:
            print("SIZE OF ARRAYS:")
            print(images.nbytes)
            print(np.array(labelsIndexes).nbytes)
            print(np.array(filenames).nbytes)
            print(np.array(typeNames).nbytes)
        except:
            print("Exception Raised")

        return typeList, images, np.array(labelsIndexes), np.array(filenames), np.array(typeNames)

    def combined_sn_gal_arrays_multiprocessing(self, snTemplateLocation, snTempFileList, galTemplateLocation, galTempFileList):
        galTempList = TempList().temp_list(galTempFileList)

        images = np.empty((0, int(self.nw)), np.float16)
        labelsIndexes = np.empty(0, np.uint16)
        filenames = np.empty(0)
        typeNames = np.empty(0)

        t1 = time.time()
        pool = mp.Pool(processes=11)
        results = [pool.apply_async(self.combined_sn_gal_templates_to_arrays, args=(snTemplateLocation, snTempFileList, galTemplateLocation, [gList],)) for gList in galTempList]
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
        except:
            print("Exception Raised")
        print("Completed Creating Arrays!")

        return typeList, images, labelsIndexes, filenames, typeNames
