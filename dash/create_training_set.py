import pickle
import numpy as np
from dash.create_arrays import AgeBinning, CreateLabels, ArrayTools, CreateArrays
import zipfile
import gzip
import os



class CreateTrainingSet(object):

    def __init__(self, snidTemplateLocation, snidTempFileList, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, galTemplateLocation=None, galTempFileList=None):
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
        self.nLabels = self.nTypes * self.numOfAgeBins
        self.createArrays = CreateArrays(w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ)
        self.arrayTools = ArrayTools(self.nLabels, self.nw)
        
    def type_amounts(self, labels):
        counts = self.arrayTools.count_labels(labels)

        return counts

    def all_templates_to_arrays(self):
        if self.galTemplateLocation is None or self.galTempFileList is None:
            snTypeList, images, labels, filenames, typeNames = self.createArrays.snid_templates_to_arrays(self.snidTemplateLocation, self.snidTempFileList)
        else:
            snTypeList, images, labels, filenames, typeNames = self.createArrays.combined_sn_gal_templates_to_arrays(self.snidTemplateLocation, self.snidTempFileList, self.galTemplateLocation, self.galTempFileList)

        imagesShuf, labelsShuf, filenamesShuf, typeNamesShuf = self.arrayTools.shuffle_arrays(images, labels, filenames, typeNames)

        typeAmounts = self.type_amounts(labels)
        
        return snTypeList, imagesShuf, labelsShuf, filenamesShuf, typeNamesShuf, typeAmounts

    def sort_data(self):
        trainPercentage = 0.8
        testPercentage = 0.2
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

        return ((trainImagesOverSample, trainLabelsOverSample, trainFilenamesOverSample, trainTypeNamesOverSample),
                (testImagesShortlist, testLabelsShortlist, testFilenamesShortlist, testTypeNamesShortlist),
                (validateImages, validateLabels, validateFilenames, validateTypeNames),
                typeAmounts)


class SaveTrainingSet(object):
    def __init__(self, snidTemplateLocation, snidTempFileList, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, galTemplateLocation=None, galTempFileList=None):
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
        self.createLabels = CreateLabels(nTypes, minAge, maxAge, ageBinSize, typeList)
        
        self.createTrainingSet = CreateTrainingSet(snidTemplateLocation, snidTempFileList, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, galTemplateLocation, galTempFileList)
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
        self.typeAmounts = self.sortData[3]

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


def create_training_set_files():
    with open('data_files/training_params.pickle', 'rb') as f1:
        pars = pickle.load(f1)
    nTypes1, w01, w11, nw1, minAge1, maxAge1, ageBinSize1, typeList1 = pars['nTypes'], pars['w0'], pars['w1'], pars['nw'], \
                                                               pars['minAge'], pars['maxAge'], pars['ageBinSize'], \
                                                               pars['typeList']

    minZ1 = 0
    maxZ1 = 0.0

    scriptDirectory = os.path.dirname(os.path.abspath(__file__))

    snidTemplateLocation1 = os.path.join(scriptDirectory, "../templates/snid_templates_Modjaz_BSNIP/")
    snidTempFileList1 = snidTemplateLocation1 + 'templist.txt'
    galTemplateLocation1 = os.path.join(scriptDirectory, "../templates/superfit_templates/gal/")
    galTempFileList1 = galTemplateLocation1 + 'gal.list'

    saveTrainingSet = SaveTrainingSet(snidTemplateLocation1, snidTempFileList1, w01, w11, nw1, nTypes1, minAge1, maxAge1, ageBinSize1, typeList1, minZ1, maxZ1)#, galTemplateLocation1, galTempFileList1)
    typeNamesList1, typeAmounts1 = saveTrainingSet.type_amounts()

    saveFilename1 = 'data_files/trainingSet_type_age_atRedshiftZero.zip'
    saveTrainingSet.save_arrays(saveFilename1)

    return saveFilename1


if __name__ == '__main__':
    trainingSetFilename = create_training_set_files()
