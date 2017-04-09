import os
import numpy as np
import zipfile
import gzip


def save_templates(saveFilename, trainImages, trainLabels, trainFilenames, trainTypeNames):
    nTypes = len(trainLabels[0])

    templateFluxesAll = []
    templateLabelsAll = []
    templateNamesAll = []
    templateFilenamesAll = []

    for c in range(nTypes):
        templateFluxesForThisType = []
        templateLabelsForThisType = []
        templateNamesForThisType = []
        templateFilenamesForThisType = []
        countTemplates = 0
        for i in range(len(trainLabels)):
            if (trainLabels[i][c] == 1):
                templateFluxesForThisType.append(trainImages[i])
                templateLabelsForThisType.append(trainLabels[i])
                templateNamesForThisType.append(trainTypeNames[i])
                templateFilenamesForThisType.append(trainFilenames[i].replace('.lnw', '').strip('_z0.0'))
                countTemplates += 1
                
        if countTemplates == 0:
            templateFluxesForThisType.append(np.zeros(len(trainImages[0])))
            templateLabelsForThisType.append(np.zeros(len(trainLabels[0])))
            templateNamesForThisType.append('No Templates')
            templateFilenamesForThisType.append('No Templates')

        templateFluxesAll.append(np.array(templateFluxesForThisType))
        templateLabelsAll.append(np.array(templateLabelsForThisType))
        templateNamesAll.append(np.array(templateNamesForThisType))
        templateFilenamesAll.append(np.array(templateFilenamesForThisType))

    # These are nTypes by numOfTemplatesForEachType dimensional arrays. Each entry in this 2D array has a 1D flux
    # They are in order of nTypes
    templateFluxesAll = np.array(templateFluxesAll)
    templateLabelsAll = np.array(templateLabelsAll)
    templateNamesAll = np.array(templateNamesAll)
    templateFilenamesAll = np.array(templateFilenamesAll)
    
    np.savez_compressed(saveFilename, templateFluxesAll=templateFluxesAll, templateLabelsAll=templateLabelsAll, templateNamesAll=templateNamesAll, templateFilenamesAll=templateFilenamesAll)


def create_template_set_file():
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    trainingSet = 'data_files/trainingSet_type_age_atRedshiftZero.zip'
    extractedFolder = 'data_files/trainingSet_type_age_atRedshiftZero'
    zipRef = zipfile.ZipFile(trainingSet, 'r')
    zipRef.extractall(extractedFolder)
    zipRef.close()

    npyFiles = {}
    for filename in os.listdir(extractedFolder):
        if filename.endswith('.gz'):
            npyFiles[filename.strip('.npy.gz')] = gzip.GzipFile(os.path.join(scriptDirectory, extractedFolder, filename), 'r')

    trainImages = np.load(npyFiles['trainImages'])
    trainLabels = np.load(npyFiles['trainLabels'])
    trainFilenames = np.load(npyFiles['trainFilenames'])
    trainTypeNames = np.load(npyFiles['trainTypeNames'])

    saveFilename = 'data_files/templates.npz'

    print("Saving Templates...")
    save_templates(saveFilename, trainImages, trainLabels, trainFilenames, trainTypeNames)
    print("Saved Templates to: " + saveFilename)

    loaded = np.load(os.path.join(scriptDirectory, saveFilename))

    return saveFilename


if __name__ == '__main__':
    templateSetFilename = create_template_set_file()
