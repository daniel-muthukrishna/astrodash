import os
import numpy as np

def save_templates():
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

if __name__ == '__main__':
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    trainingSet = "type_age_atRedshiftZero.npz"
    print("Reading " + trainingSet + " ...")
    loaded = np.load(os.path.join(scriptDirectory, trainingSet))
    trainImages = loaded['trainImages']
    trainLabels = loaded['trainLabels']
    trainFilenames = loaded['trainFilenames']
    trainTypeNames = loaded['trainTypeNames']

    saveFilename = 'templates.npz'

    print("Saving Templates...")
    save_templates()
    print("Saved Templates to: " + saveFilename)

    loaded = np.load(os.path.join(scriptDirectory, saveFilename))
