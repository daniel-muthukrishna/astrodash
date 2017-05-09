import pickle
import numpy as np
from dash.create_arrays import AgeBinning, CreateLabels, ArrayTools, CreateArrays
import zipfile
import gzip
import os


class CreateTrainingSet(object):

    def __init__(self, snidTemplateLocation, snidTempFileList, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, redshiftPrecision, galTemplateLocation, galTempFileList, hostTypes, nHostTypes):
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
        self.ageBinning = AgeBinning(self.minAge, self.maxAge, self.ageBinSize)
        self.numOfAgeBins = self.ageBinning.age_bin(self.maxAge-0.1) + 1
        self.nLabels = self.nTypes * self.numOfAgeBins * nHostTypes
        self.createArrays = CreateArrays(w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, redshiftPrecision, hostTypes, nHostTypes)
        self.arrayTools = ArrayTools(self.nLabels, self.nw)
        
    def type_amounts(self, labels):
        counts = self.arrayTools.count_labels(labels)

        return counts

    def all_templates_to_arrays(self):
        if self.galTemplateLocation is None or self.galTempFileList is None:
            snTypeList, images, labels, filenames, typeNames = self.createArrays.snid_templates_to_arrays(self.snidTemplateLocation, self.snidTempFileList)
        else:
            snTypeList, images, labels, filenames, typeNames = self.createArrays.combined_sn_gal_arrays_multiprocessing(self.snidTemplateLocation, self.snidTempFileList, self.galTemplateLocation, self.galTempFileList)

        imagesShuf, labelsShuf, filenamesShuf, typeNamesShuf = self.arrayTools.shuffle_arrays(images, labels, filenames, typeNames)

        typeAmounts = self.type_amounts(labels)
        
        return snTypeList, imagesShuf, labelsShuf, filenamesShuf, typeNamesShuf, typeAmounts

    def sort_data(self):
        trainPercentage = 0.9
        testPercentage = 0.1
        validatePercentage = 0.

        typeList, images, labels, filenames, typeNames, typeAmounts = self.all_templates_to_arrays()

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

        typeAmountsOverSampled = self.type_amounts(trainLabelsOverSample)

        return ((trainImagesOverSample, trainLabelsOverSample, trainFilenamesOverSample, trainTypeNamesOverSample),
                (testImagesShortlist, testLabelsShortlist, testFilenamesShortlist, testTypeNamesShortlist),
                (validateImages, validateLabels, validateFilenames, validateTypeNames),
                (typeAmounts, typeAmountsOverSampled))


class SaveTrainingSet(object):
    def __init__(self, snidTemplateLocation, snidTempFileList, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, redshiftPrecision, galTemplateLocation=None, galTempFileList=None, hostTypes=None, nHostTypes=1):
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
        
        self.createTrainingSet = CreateTrainingSet(snidTemplateLocation, snidTempFileList, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, redshiftPrecision, galTemplateLocation, galTempFileList, hostTypes, nHostTypes)
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
        self.typeAmounts = self.sortData[3][0]
        self.typeAmountsOverSampled = self.sortData[3][1]

        self.typeNamesList = self.createLabels.type_names_list()

    def type_amounts(self):
        for i in range(len(self.typeNamesList)):
            print(str(self.typeAmounts[i]) + ": " + str(self.typeNamesList[i]))
        return self.typeNamesList, self.typeAmounts

    def save_arrays(self, saveFilename):
        saveFilename = saveFilename
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


def create_training_set_files(dataDirName, minZ=0, maxZ=0, redshiftPrecision=0.01, trainWithHost=True, classifyHost=False):
    with open(dataDirName + 'training_params.pickle', 'rb') as f1:
        pars = pickle.load(f1)
    nTypes, w0, w1, nw, minAge, maxAge, ageBinSize, typeList = pars['nTypes'], pars['w0'], pars['w1'], \
                                                                         pars['nw'], pars['minAge'], pars['maxAge'], \
                                                                         pars['ageBinSize'], pars['typeList']
    hostList, nHostTypes = None, 1

    scriptDirectory = os.path.dirname(os.path.abspath(__file__))

    snidTemplateLocation = os.path.join(scriptDirectory, "../templates/snid_templates_Modjaz_BSNIP/")
    snidTempFileList = snidTemplateLocation + 'templist.txt'
    if trainWithHost:
        galTemplateLocation = os.path.join(scriptDirectory, "../templates/superfit_templates/gal/")
        galTempFileList = galTemplateLocation + 'gal.list'
        if classifyHost:
            hostList = pars['galTypeList']
            nHostTypes = len(hostList)
    else:
        galTemplateLocation, galTempFileList = None, None

    saveTrainingSet = SaveTrainingSet(snidTemplateLocation, snidTempFileList, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, redshiftPrecision, galTemplateLocation, galTempFileList, hostList, nHostTypes)
    typeNamesList, typeAmounts = saveTrainingSet.type_amounts()

    saveFilename = dataDirName + 'training_set.zip'
    saveTrainingSet.save_arrays(saveFilename)

    return saveFilename


if __name__ == '__main__':
    trainingSetFilename = create_training_set_files('data_files/', minZ=0, maxZ=0, redshiftPrecision=0.01, trainWithHost=False, classifyHost=False)
