import numpy as np
import pickle
import os
from astrodash.create_arrays import AgeBinning
from astrodash.helpers import temp_list
from astrodash.combine_sn_and_host import BinTemplate


def create_sn_and_host_arrays(snTemplateDirectory, snTempFileList, galTemplateDirectory, galTempFileList, paramsFile):
    snTemplates = {}
    galTemplates = {}
    snList = temp_list(snTempFileList)
    galList = temp_list(galTempFileList)
    with open(paramsFile, 'rb') as f:
        pars = pickle.load(f)
    w0, w1, nw, snTypes, galTypes, minAge, maxAge, ageBinSize = pars['w0'], pars['w1'], pars['nw'], pars['typeList'], \
                                                                pars['galTypeList'], pars['minAge'], pars['maxAge'], \
                                                                pars['ageBinSize']
    ageBinning = AgeBinning(minAge, maxAge, ageBinSize)
    ageLabels = ageBinning.age_labels()
    # Create dictionary of dictionaries for type and age of SN
    for snType in snTypes:
        snTemplates[snType] = {}
        for ageLabel in ageLabels:
            snTemplates[snType][ageLabel] = {}
            snTemplates[snType][ageLabel]['snInfo'] = []
            snTemplates[snType][ageLabel]['names'] = []
    for galType in galTypes:
        galTemplates[galType] = {}
        galTemplates[galType]['galInfo'] = []
        galTemplates[galType]['names'] = []

    for snFile in snList:
        snBinTemplate = BinTemplate(snTemplateDirectory + snFile, 'sn', w0, w1, nw)
        nAges = snBinTemplate.nCols
        ages = snBinTemplate.ages
        snType = snBinTemplate.tType
        filename = snBinTemplate.filename
        for ageIdx in range(nAges):
            age = ages[ageIdx]
            if minAge < age < maxAge:
                ageBin = ageBinning.age_bin(age)
                ageLabel = ageLabels[ageBin]
                snInfo = snBinTemplate.bin_template(ageIdx)
                snTemplates[snType][ageLabel]['snInfo'].append(snInfo)
                snTemplates[snType][ageLabel]['names'].append("%s_%s" % (filename, age))

            print("Reading {} {} out of {}".format(snFile, ageIdx, nAges))

    for galFile in galList:
        galBinTemplate = BinTemplate(galTemplateDirectory + galFile, 'gal', w0, w1, nw)
        galType = galBinTemplate.tType
        filename = galBinTemplate.filename
        galInfo = galBinTemplate.bin_template()
        galTemplates[galType]['galInfo'].append(galInfo)
        galTemplates[galType]['names'].append(filename)

        print("Reading {}".format(galFile))

    # Convert lists in dictionaries to numpy arrays
    for snType in snTypes:
        for ageLabel in ageLabels:
            snTemplates[snType][ageLabel]['snInfo'] = np.array(snTemplates[snType][ageLabel]['snInfo'])
            snTemplates[snType][ageLabel]['names'] = np.array(snTemplates[snType][ageLabel]['names'])
    for galType in galTypes:
        galTemplates[galType]['galInfo'] = np.array(galTemplates[galType]['galInfo'])
        galTemplates[galType]['names'] = np.array(galTemplates[galType]['names'])

    return snTemplates, galTemplates


def save_templates():
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    parameterFile = 'models_v06/models/zeroZ/training_params.pickle'
    snTemplateDirectory = os.path.join(scriptDirectory, "../templates/training_set/")
    snTempFileList = snTemplateDirectory + 'templist.txt'
    galTemplateDirectory = os.path.join(scriptDirectory, "../templates/superfit_templates/gal/")
    galTempFileList = galTemplateDirectory + 'gal.list'
    saveFilename = 'models_v06/models/sn_and_host_templates.npz'

    snTemplates, galTemplates = create_sn_and_host_arrays(snTemplateDirectory, snTempFileList, galTemplateDirectory, galTempFileList, parameterFile)

    np.savez_compressed(saveFilename, snTemplates=snTemplates, galTemplates=galTemplates)

    return saveFilename


if __name__ == "__main__":
    unCombinedTemplates = save_templates()
