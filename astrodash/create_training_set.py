import pickle
import numpy as np
import zipfile
import gzip
import os
import random
import copy
from collections import OrderedDict
from astrodash.create_arrays import AgeBinning, CreateLabels, ArrayTools, CreateArrays
from astrodash.helpers import temp_list

random.seed(42)


class CreateTrainingSet(object):

    def __init__(self, snidTemplateLocation, snidTempFileList, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, numOfRedshifts, galTemplateLocation, galTempFileList, hostTypes, nHostTypes, trainFraction):
        self.snidTemplateLocation = snidTemplateLocation
        self.snidTempFileList = snidTempFileList
        self.galTemplateLocation = galTemplateLocation
        self.galTempFileList = galTempFileList
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize
        self.typeList = typeList
        self.trainFraction = trainFraction
        self.ageBinning = AgeBinning(self.minAge, self.maxAge, self.ageBinSize)
        self.numOfAgeBins = self.ageBinning.age_bin(self.maxAge-0.1) + 1
        self.nLabels = self.nTypes * self.numOfAgeBins * nHostTypes
        self.createArrays = CreateArrays(w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, numOfRedshifts, hostTypes, nHostTypes)
        self.arrayTools = ArrayTools(self.nLabels, self.nw)
        
    def type_amounts(self, labels):
        counts = self.arrayTools.count_labels(labels)

        return counts

    def all_templates_to_arrays(self, snTempFileList, galTemplateLocation):
        """
        Parameters
        ----------
        snTempFileList : list or dictionary
        galTemplateLocation

        Returns
        -------
        """
        images, labels, filenames, typeNames = self.createArrays.combined_sn_gal_arrays_multiprocessing(self.snidTemplateLocation, snTempFileList, galTemplateLocation, self.galTempFileList)

        arraysShuf = self.arrayTools.shuffle_arrays(images=images, labels=labels, filenames=filenames, typeNames=typeNames, memmapName='all')

        typeAmounts = self.type_amounts(labels)
        
        return arraysShuf, typeAmounts

    def train_test_split(self):
        """
        Split training set before creating arrays.
        Maybe should change this to include ages in train/test split instead of just SN files.
        """
        snTempFileList = copy.copy(self.snidTempFileList)
        fileList = temp_list(snTempFileList)
        snAndAgeIdxDict = OrderedDict()
        spectraList = []

        # SPLIT BY SPECTRA
        # Get number of spectra per file
        for i, sn in enumerate(fileList):
            with open(os.path.join(self.snidTemplateLocation, sn), 'r') as FileObj:
                for lineNum, line in enumerate(FileObj):
                    # Read Header Info
                    if lineNum == 0:
                        header = (line.strip('\n')).split(' ')
                        header = [x for x in header if x != '']
                        numAges, nwx, w0x, w1x, mostKnots, tname, dta, ttype, ittype, itstype = header
                        numAges, mostKnots = map(int, (numAges, mostKnots))
                    elif lineNum == mostKnots + 2:
                        ages = np.array(line.split()[1:]).astype(float)
                        agesIndexesInRange = np.where((ages >= self.minAge) & (ages <= self.maxAge))[0]
                        snAndAgeIdxDict[sn] = agesIndexesInRange
                        for ageIdx in agesIndexesInRange:
                            spectraList.append((sn, ageIdx))

        # Split train/test
        random.shuffle(spectraList)
        trainSize = int(self.trainFraction * len(spectraList))
        trainSpectra = spectraList[:trainSize]
        testSpectra = spectraList[trainSize:]

        trainDict, testDict = OrderedDict(), OrderedDict()
        for k, v in trainSpectra:
            trainDict.setdefault(k, []).append(v)
        for k, v in testSpectra:
            testDict.setdefault(k, []).append(v)

        # # SPLIT BY FILENAME INSTEAD OF BY SPECTRA
        # random.Random(42).shuffle(fileList)
        #
        # trainSize = int(self.trainFraction * len(fileList))
        # dirName = os.path.dirname(self.snidTempFileList)
        # trainListFileName = os.path.join(dirName, 'train_templist.txt')
        # testListFileName = os.path.join(dirName, 'test_templist.txt')
        #
        # # Save train set file list
        # with open(trainListFileName, 'w') as f:
        #     for line in fileList[:trainSize]:
        #         f.write("%s\n" % line)
        #
        # # Save test set file list
        # with open(testListFileName, 'w') as f:
        #     for line in fileList[trainSize:]:
        #         f.write("%s\n" % line)
        print("trainDict", trainDict)
        print("testDict", testDict)
        return trainDict, testDict

    def sort_data(self):
        if self.trainFraction == 1.0:
            arrays, typeAmounts = self.all_templates_to_arrays(self.snidTempFileList, self.galTemplateLocation)
            trainImages, trainLabels, trainFilenames, trainTypeNames = arrays['images'], arrays['labels'], arrays['filenames'], arrays['typeNames']
            testImages, testLabels, testFilenames, testTypeNames = trainImages[-1:], trainLabels[-1:], trainFilenames[-1:], trainTypeNames[-1:]
        else:
            trainDict, testDict = self.train_test_split()

            arraysTrain, typeAmountsTrain = self.all_templates_to_arrays(trainDict, self.galTemplateLocation)
            trainImages, trainLabels, trainFilenames, trainTypeNames = arraysTrain['images'], arraysTrain['labels'], arraysTrain['filenames'], arraysTrain['typeNames']

            arraysTest, typeAmountsTest = self.all_templates_to_arrays(testDict, None)
            testImages, testLabels, testFilenames, testTypeNames = arraysTest['images'], arraysTest['labels'], arraysTest['filenames'], arraysTest['typeNames']

        # trainPercentage = self.trainFraction
        # testPercentage = 1.0 - self.trainFraction
        #
        # arrays, typeAmounts = self.all_templates_to_arrays(self.snidTempFileList, self.galTemplateLocation)
        # images, labels, filenames, typeNames = arrays['images'], arrays['labels'], arrays['filenames'], arrays['typeNames']
        #
        # trainSize = int(trainPercentage * len(images))
        # testSize = int(testPercentage * len(images))
        #
        # trainImages = images[:trainSize]
        # testImages = images[trainSize: trainSize + testSize]
        # trainLabels = labels[:trainSize]
        # testLabels = labels[trainSize: trainSize + testSize]
        # trainFilenames = filenames[:trainSize]
        # testFilenames = filenames[trainSize: trainSize + testSize]
        # trainTypeNames = typeNames[:trainSize]
        # testTypeNames = typeNames[trainSize: trainSize + testSize]

        typeAmounts = self.type_amounts(trainLabels)

        return ((trainImages, trainLabels, trainFilenames, trainTypeNames),
                (testImages, testLabels, testFilenames, testTypeNames),
                typeAmounts)


class SaveTrainingSet(object):
    def __init__(self, snidTemplateLocation, snidTempFileList, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, numOfRedshifts, galTemplateLocation=None, galTempFileList=None, hostTypes=None, nHostTypes=1, trainFraction=0.8):
        self.snidTemplateLocation = snidTemplateLocation
        self.snidTempFileList = snidTempFileList
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize
        self.typeList = typeList
        self.createLabels = CreateLabels(nTypes, minAge, maxAge, ageBinSize, typeList, hostTypes, nHostTypes)
        
        self.createTrainingSet = CreateTrainingSet(snidTemplateLocation, snidTempFileList, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, numOfRedshifts, galTemplateLocation, galTempFileList, hostTypes, nHostTypes, trainFraction)
        self.sortData = self.createTrainingSet.sort_data()
        self.trainImages = self.sortData[0][0]
        self.trainLabels = self.sortData[0][1]
        self.trainFilenames = self.sortData[0][2]
        self.trainTypeNames = self.sortData[0][3]
        self.testImages = self.sortData[1][0]
        self.testLabels = self.sortData[1][1]
        self.testFilenames = self.sortData[1][2]
        self.testTypeNames = self.sortData[1][3]
        self.typeAmounts = self.sortData[2]

        self.typeNamesList = self.createLabels.type_names_list()

    def type_amounts(self):
        for i in range(len(self.typeNamesList)):
            print(str(self.typeAmounts[i]) + ": " + str(self.typeNamesList[i]))
        return self.typeNamesList, self.typeAmounts

    def save_arrays(self, saveFilename):
        arraysToSave = {'trainImages.npy.gz': self.trainImages, 'trainLabels.npy.gz': self.trainLabels,
                        'testImages.npy.gz': self.testImages, 'testLabels.npy.gz': self.testLabels,
                        'testTypeNames.npy.gz': self.testTypeNames, 'typeNamesList.npy.gz': self.typeNamesList,
                        'trainFilenames.npy.gz': self.trainFilenames, 'trainTypeNames.npy.gz': self.trainTypeNames}

        try:
            print("SIZE OF ARRAYS TRAINING:")
            print(self.trainImages.nbytes, self.testImages.nbytes)
            print(self.trainLabels.nbytes, self.testLabels.nbytes)
            print(self.trainFilenames.nbytes, self.testFilenames.nbytes)
            print(self.trainTypeNames.nbytes, self.testTypeNames.nbytes)
        except:
            print("Exception Raised")

        for filename, array in arraysToSave.items():
            f = gzip.GzipFile(filename, "w")
            np.save(file=f, arr=array)
            f.close()

        with zipfile.ZipFile(saveFilename, 'w') as myzip:
            for f in arraysToSave.keys():
                myzip.write(f)

        print("Saved Training Set to: " + saveFilename)

        # Delete npy.gz files
        for filename in arraysToSave.keys():
            os.remove(filename)


def create_training_set_files(dataDirName, minZ=0, maxZ=0, numOfRedshifts=80, trainWithHost=True, classifyHost=False, trainFraction=0.8):
    with open(os.path.join(dataDirName, 'training_params.pickle'), 'rb') as f1:
        pars = pickle.load(f1)
    nTypes, w0, w1, nw, minAge, maxAge, ageBinSize, typeList = pars['nTypes'], pars['w0'], pars['w1'], \
                                                                         pars['nw'], pars['minAge'], pars['maxAge'], \
                                                                         pars['ageBinSize'], pars['typeList']
    hostList, nHostTypes = None, 1

    scriptDirectory = os.path.dirname(os.path.abspath(__file__))

    snidTemplateLocation = os.path.join(scriptDirectory, "../templates/training_set/")
    snidTempFileList = snidTemplateLocation + 'templist.txt'
    if trainWithHost:
        galTemplateLocation = os.path.join(scriptDirectory, "../templates/superfit_templates/gal/")
        galTempFileList = galTemplateLocation + 'gal.list'
        if classifyHost:
            hostList = pars['galTypeList']
            nHostTypes = len(hostList)
    else:
        galTemplateLocation, galTempFileList = None, None

    saveTrainingSet = SaveTrainingSet(snidTemplateLocation, snidTempFileList, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, numOfRedshifts, galTemplateLocation, galTempFileList, hostList, nHostTypes, trainFraction)
    typeNamesList, typeAmounts = saveTrainingSet.type_amounts()

    saveFilename = os.path.join(dataDirName, 'training_set.zip')
    saveTrainingSet.save_arrays(saveFilename)

    return saveFilename


if __name__ == '__main__':
    trainingSetFilename = create_training_set_files('data_files/', minZ=0, maxZ=0, numOfRedshifts=80, trainWithHost=False, classifyHost=False, trainFraction=0.8)


# # Split by filename instead of by spectra
# snTempFileList = copy.copy(self.snidTempFileList)
# fileList = temp_list(snTempFileList)
# random.Random(42).shuffle(fileList)
#
# trainSize = int(self.trainFraction * len(fileList))
# dirName = os.path.dirname(self.snidTempFileList)
# trainListFileName = os.path.join(dirName, 'train_templist.txt')
# testListFileName = os.path.join(dirName, 'test_templist.txt')
#
# # Save train set file list
# with open(trainListFileName, 'w') as f:
#     for line in fileList[:trainSize]:
#         f.write("%s\n" % line)
#
# # Save test set file list
# with open(testListFileName, 'w') as f:
#     for line in fileList[trainSize:]:
#         f.write("%s\n" % line)
