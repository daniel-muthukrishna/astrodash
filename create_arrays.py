import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sn_processing import PreProcessing


class AgeBinning(object):
    def __init__(self, minAge, maxAge, ageBinSize):
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize

    def age_bin(self, age):
        ageBin = int(age / self.ageBinSize) - int(self.minAge / self.ageBinSize)  # around zero has double bin siz

        return ageBin

    def age_labels(self):
        ageLabels = []

        ageBinPrev = 0
        ageLabelMin = self.minAge
        for age in np.arange(self.minAge, self.maxAge):
            ageBin = self.age_bin(age)

            if (ageBin != ageBinPrev):
                ageLabelMax = age
                ageLabels.append(str(ageLabelMin) + " to " + str(ageLabelMax))
                ageLabelMin = ageLabelMax

            ageBinPrev = ageBin

        ageLabels.append(str(ageLabelMin) + " to " + str(maxAge))

        return ageLabels


class CreateLabels(object):

    def __init__(self, nTypes, minAge, maxAge, ageBinSize):
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize
        self.ageBinning = AgeBinning(self.minAge, self.maxAge, self.ageBinSize)
        self.numOfAgeBins = self.ageBinning.age_bin(self.maxAge) + 1
        self.nLabels = self.nTypes * self.numOfAgeBins
        self.ageLabels = self.ageBinning.age_labels()



    def label_array(self, ttype, age):
        ageBin = self.ageBinning.age_bin(age)
        labelarray = np.zeros((self.nTypes, self.numOfAgeBins))
        typeNames = []

        if (ttype == 'Ia-norm'):
            typeIndex = 0

        elif (ttype == 'IIb'):
            typeIndex = 1

        elif (ttype == 'Ia-pec'):
            typeIndex = 2

        elif (ttype == 'Ic-broad'):
            typeIndex = 3

        elif (ttype == 'Ia-csm'):
            typeIndex = 4

        elif (ttype == 'Ic-norm'):
            typeIndex = 5

        elif (ttype == 'IIP'):
            typeIndex = 6

        elif (ttype == 'Ib-pec'):
            typeIndex = 7

        elif (ttype == 'IIL'):
            typeIndex = 8

        elif (ttype == 'Ib-norm'):
            typeIndex = 9

        elif (ttype == 'Ia-91bg'):
            typeIndex = 10

        elif (ttype == 'II-pec'):
            typeIndex = 11

        elif (ttype == 'Ia-91T'):
            typeIndex = 12

        elif (ttype == 'IIn'):
            typeIndex = 13

        elif (ttype == 'Ia'):
            typeIndex = 0

        elif (ttype == 'Ib'):
            typeIndex = 9

        elif (ttype == 'Ic'):
            typeIndex = 5

        elif (ttype == 'II'):
            typeIndex = 13
        else:
            print ("Invalid type")

        labelarray[typeIndex][ageBin] = 1
        labelarray = labelarray.flatten()

        typeNames.append(ttype + ": " + self.ageLabels[ageBin])
        typeNames = np.array(typeNames)

        return labelarray, typeNames


class ReadSpectra(object):

    def __init__(self, w0, w1, nw):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw


    def temp_list(self, tempFileList):
        f = open(tempFileList, 'rU')

        fileList = f.readlines()
        for i in range(0,len(fileList)):
            fileList[i] = fileList[i].strip('\n')

        f.close()

        return fileList

    def snid_template_data(self, snidTemplateLocation, filename, ageIdx):
        """ lnw template files """
        data = PreProcessing()
        wave, flux, nCols, ages, tType, minIndex, maxIndex = data.templates(snidTemplateLocation+filename, ageIdx, self.w0, self.w1, self.nw)

        return wave, flux, nCols, ages, tType, minIndex, maxIndex


    def sf_age(self, filename):
        snName, extension = filename.strip('.dat').split('.')
        ttype, snName = snName.split('/')

        if (extension == 'max'):
            age = 0
        elif (extension[0] == 'p'):
            age = float(extension[1:])
        elif (extension[0] == 'm'):
            age = -float(extension[1:])
        else:
            print "Invalid Superfit Filename: " + filename

        return snName, ttype, age


    def superfit_template_data(self, sfTemplateLocation, filename):
        """ Returns wavelength and flux after all preprocessing """
        data = PreProcessing()
        wave, flux, minIndex, maxIndex = data.processed_data(sfTemplateLocation + filename, w0, w1, nw)
        snName, ttype, age = self.sf_age(filename)

        print snName, ttype, age

        return wave, flux, minIndex, maxIndex, age, snName, ttype



class ArrayTools(object):

    def __init__(self, nLabels):
        self.nLabels = nLabels

    def shuffle_arrays(self, images, labels, filenames, typeNames):
        imagesShuf = []
        labelsShuf = []
        filenamesShuf = []
        typeNamesShuf = []

        # Randomise order
        indexShuf = range(len(images))
        shuffle(indexShuf)
        for i in indexShuf:
            imagesShuf.append(images[i])
            labelsShuf.append(labels[i])
            filenamesShuf.append(filenames[i])
            typeNamesShuf.append(typeNames[i])

        return np.array(imagesShuf), np.array(labelsShuf), np.array(filenamesShuf), np.array(typeNamesShuf)


    def count_labels(self, labels):
        counts = np.zeros(self.nLabels)

        for i in range(len(labels)):
            counts = labels[i] + counts

        return counts


    def div0(self, a, b):
        """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[~ np.isfinite(c)] = 0  # -inf inf NaN
        return c


    def over_sample_arrays(self, images, labels, filenames, typeNames):
        counts = self.count_labels(labels)
        print "Before OverSample"  #
        print counts  #

        overSampleAmount = self.div0(5 * max(counts), counts)  # ignore zeros in counts
        imagesOverSampled = []
        labelsOverSampled = []
        filenamesOverSampled = []
        typeNamesOverSampled = []

        counts1 = np.zeros(self.nLabels)

        imagesShuf, labelsShuf, filenamesShuf, typeNamesShuf = self.shuffle_arrays(images, labels, filenames, typeNames)

        for i in range(len(labelsShuf)):
            label = labelsShuf[i]
            image = imagesShuf[i]
            filename = filenamesShuf[i]
            typeName = typeNamesShuf[i]

            labelIndex = np.argmax(label)

            for r in range(int(overSampleAmount[labelIndex])):
                imagesOverSampled.append(image)
                labelsOverSampled.append(label)
                filenamesOverSampled.append(filename)
                typeNamesOverSampled.append(typeName)
                counts1 = label + counts1
        print "After OverSample"  #
        print counts1  #

        # [ 372.    8.   22.   12.    1.   22.   26.    6.    1.    7.   34.    5.  44.    6.]
        imagesOverSampled = np.array(imagesOverSampled)
        labelsOverSampled = np.array(labelsOverSampled)
        filenamesOverSampled = np.array(filenamesOverSampled)
        typeNamesOverSampled = np.array(typeNamesOverSampled)
        imagesOverSampledShuf, labelsOverSampledShuf, filenamesOverSampledShuf, typeNamesOverSampledShuf = self.shuffle_arrays(imagesOverSampled, labelsOverSampled, filenamesOverSampled, typeNamesOverSampled)

        return imagesOverSampledShuf, labelsOverSampledShuf, filenamesOverSampledShuf, typeNamesOverSampledShuf


class CreateArrays(object):
    def __init__(self, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize):
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
        self.readSpectra = ReadSpectra(self.w0, self.w1, self.nw)
        self.createLabels = CreateLabels(self.nTypes, self.minAge, self.maxAge, self.ageBinSize)


    def snid_templates_to_arrays(self, snidTemplateLocation, tempfilelist):
        ''' This function is for the SNID processed files, which
            have been preprocessed to negatives, and so cannot be
            imaged yet '''
        templist = self.readSpectra.temp_list(tempfilelist)
        typeList = []
        images = np.empty((0, self.nw), np.float32)  # Number of pixels
        labels = np.empty((0, self.nLabels), float)  # Number of labels (SN types)
        filenames = []
        typeNames = []
        agesList = []

        for i in range(0, len(templist)):
            ncols = 15
            for ageidx in range(0, 100):
                if (ageidx < ncols):
                    tempwave, tempflux, ncols, ages, ttype, tminindex, tmaxindex = self.readSpectra.snid_template_data(snidTemplateLocation, templist[i], ageidx)
                    agesList.append(ages[ageidx])

                    if ((float(ages[ageidx]) > minAge and float(ages[ageidx]) < maxAge)):
                        label, typeName = self.createLabels.label_array(ttype, ages[ageidx])
                        nonzeroflux = tempflux[tminindex:tmaxindex + 1]
                        newflux = (nonzeroflux - min(nonzeroflux)) / (max(nonzeroflux) - min(nonzeroflux))
                        newflux2 = np.concatenate((tempflux[0:tminindex], newflux, tempflux[tmaxindex + 1:]))
                        images = np.append(images, np.array([newflux2]), axis=0)  # images.append(newflux2)
                        labels = np.append(labels, np.array([label]), axis=0)  # labels.append(ttype)
                        filenames.append(templist[i] + '_' + ttype + '_' + str(ages[ageidx]))
                        typeNames.append(typeName)

            print templist[i]
            # Create List of all SN types
            if ttype not in typeList:
                typeList.append(ttype)

        return typeList, images, labels, np.array(filenames), typeNames

    def superfit_templates_to_arrays(self, sfTemplateLocation, sftempfilelist):
        templist = self.readSpectra.temp_list(sftempfilelist)
        images = np.empty((0, self.nw), np.float32)  # Number of pixels
        labels = np.empty((0, self.nLabels), float)  # Number of labels (SN types)
        filenames = []
        typeNames = []

        for i in range(0, len(templist)):
            tempwave, tempflux, tminindex, tmaxindex, age, snName, ttype = self.readSpectra.superfit_template_data(
                sfTemplateLocation, templist[i])

            if ((float(ages[ageidx]) > minAge and float(ages[ageidx]) > maxAge)):
                label, typeName = label_array(ttype, ages[ageidx])
                nonzeroflux = tempflux[tminindex:tmaxindex + 1]
                newflux = (nonzeroflux - min(nonzeroflux)) / (max(nonzeroflux) - min(nonzeroflux))
                newflux2 = np.concatenate((tempflux[0:tminindex], newflux, tempflux[tmaxindex + 1:]))
                images = np.append(images, np.array([newflux2]), axis=0)  # images.append(newflux2)
                labels = np.append(labels, np.array([label]), axis=0)  # labels.append(ttype)
                filenames.append(templist[i])
                typeNames.append(typeName)

        return images, labels, np.array(filenames), typeNames


class CreateTrainingSet(object):

    def __init__(self, snidTemplateLocation, snidtempfilelist, sfTemplateLocation, sftempfilelist, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize):
        self.snidTemplateLocation = snidTemplateLocation
        self.snidtempfilelist = snidtempfilelist
        self.sfTemplateLocation = sfTemplateLocation
        self.sftempfilelist = sftempfilelist
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
        self.createArrays = CreateArrays(self.w0, self.w1, self.nw, self.nTypes, self.minAge, self.maxAge, self.ageBinSize)
        self.arrayTools = ArrayTools(self.nLabels)


    def all_templates_to_arrays(self):
        images = np.empty((0, self.nw), np.float32)  # Number of pixels
        labels = np.empty((0, self.nTypes), float)  # Number of labels (SN types)
        typeList = []

        typelistSnid, imagesSnid, labelsSnid, filenamesSnid, typeNamesSnid = self.createArrays.snid_templates_to_arrays(self.snidTemplateLocation, self.snidtempfilelist)
        # imagesSuperfit, labelsSuperfit, filenamesSuperfit, typeNamesSuperfit = superfit_templates_to_arrays(self.sfTemplateLocation, sftempfilelist)

        images = np.vstack((imagesSnid))  # , imagesSuperfit)) #Add in other templates from superfit etc.
        labels = np.vstack((labelsSnid))  # , labelsSuperfit))
        filenames = np.hstack((filenamesSnid))  # , filenamesSuperfit))
        typeNames = np.hstack((typeNamesSnid))

        typeList = typelistSnid

        imagesShuf, labelsShuf, filenamesShuf, typeNamesShuf = self.arrayTools.shuffle_arrays(images, labels, filenames, typeNames)  # imagesShortlist, labelsShortlist = shortlist_arrays(images, labels)

        return typeList, imagesShuf, labelsShuf, filenamesShuf, typeNamesShuf  # imagesShortlist, labelsShortlist


    def sort_data(self):
        trainPercentage = 0.8
        testPercentage = 0.2
        validatePercentage = 0.

        typeList, images, labels, filenames, typeNames = self.all_templates_to_arrays()
        # imagesSuperfit, labelsSuperfit, filenamesSuperfit = superfit_templates_to_arrays(sftempfilelist)
        # imagesSuperfit, labelsSuperfit, filenamesSuperfit = shuffle_arrays(imagesSuperfit, labelsSuperfit, filenamesSuperfit)


        trainSize = int(trainPercentage * len(images))
        testSize = int(testPercentage * len(images))

        trainImages = images[:trainSize]
        testImages = images[trainSize: trainSize + testSize]
        validateImages = images[trainSize + testSize:]
        trainLabels = labels[:trainSize]
        testLabels = labels[trainSize: trainSize + testSize]
        validateLabels = labels[trainSize + testSize:]
        trainFilenames = filenames[:trainSize]
        testFilenames = filenames[trainSize: trainSize + testSize]
        validateFilenames = filenames[trainSize + testSize:]
        trainTypeNames = typeNames[:trainSize]
        testTypeNames = typeNames[trainSize: trainSize + testSize]
        validateTypeNames = typeNames[trainSize + testSize:]

        trainImagesOverSample, trainLabelsOverSample, trainFilenamesOverSample, trainTypeNamesOverSample = self.arrayTools.over_sample_arrays(trainImages, trainLabels, trainFilenames, trainTypeNames)
        testImagesShortlist, testLabelsShortlist, testFilenamesShortlist, testTypeNamesShortlist = testImages, testLabels, testFilenames, testTypeNames  # (testImages, testLabels, testFilenames)

        return ((trainImagesOverSample, trainLabelsOverSample, trainFilenamesOverSample, trainTypeNamesOverSample),
                (testImagesShortlist, testLabelsShortlist, testFilenamesShortlist, testTypeNamesShortlist),
                (validateImages, validateLabels, validateFilenames, validateTypeNames))


class SaveTrainingSet(object):
    def __init__(self, snidTemplateLocation, snidtempfilelist, sfTemplateLocation, sftempfilelist, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize):
        self.snidTemplateLocation = snidTemplateLocation
        self.snidtempfilelist = snidtempfilelist
        self.sfTemplateLocation = sfTemplateLocation
        self.sftempfilelist = sftempfilelist
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize

        self.createTrainingSet = CreateTrainingSet(self.snidTemplateLocation, self.snidtempfilelist, self.sfTemplateLocation, self.sftempfilelist, self.w0, self.w1, self.nw, self.nTypes, self.minAge, self.maxAge, self.ageBinSize)
        self.sortData = self.createTrainingSet.sort_data()
        self.trainImages = self.sortData[0][0]
        self.trainLabels = self.sortData[0][1]
        self.trainFilenames = self.sortData[0][2]
        self.trainTypeNames = self.sortData[0][3]
        self.testImages = self.sortData[1][0]
        self.testLabels = self.sortData[1][1]
        self.testFilenames = self.sortData[1][2]
        self.testTypeNames = self.sortData[1][3]
        self.validateImages = self.sortData[2][0]
        self.validateLabels = self.sortData[2][1]
        self.validateFilenames = self.sortData[2][2]
        self.validateTypeNames = self.sortData[2][3]

        self.save_arrays()

    def save_arrays(self):
        np.savez_compressed('file_w_ages.npz', trainImages=self.trainImages, trainLabels=self.trainLabels,
                        trainFilenames=self.trainFilenames, trainTypeNames=self.trainTypeNames,
                        testImages=self.testImages, testLabels=self.testLabels,
                        testFilenames=self.testFilenames, testTypeNames=self.testTypeNames)


nTypes = 14
w0 = 2500. #wavelength range in Angstroms
w1 = 11000.
nw = 1024. #number of wavelength bins
minAge = -50
maxAge = 50
ageBinSize = 4.
snidTemplateLocation = '/home/dan/Desktop/SNClassifying/templates/'
sfTemplateLocation = '/home/dan/Desktop/SNClassifying/templates/superfit_templates/sne/'
snidtempfilelist1 = snidTemplateLocation + 'templist'
sftempfilelist1 = sfTemplateLocation + 'templist.txt'

SaveTrainingSet(snidTemplateLocation, snidtempfilelist1, sfTemplateLocation, sftempfilelist1, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize)
