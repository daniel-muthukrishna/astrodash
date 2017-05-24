import numpy as np
import pickle
import os
from dash.preprocessing import ReadSpectrumFile, ProcessingTools, PreProcessSpectrum
from dash.create_arrays import TempList, AgeBinning
from dash.array_tools import zero_non_overlap_part, normalise_spectrum


class BinTemplate(object):
    def __init__(self, filename, templateType, w0, w1, nw):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.filename = filename.split('/')[-1]
        self.templateType = templateType
        self.preProcess = PreProcessSpectrum(w0, w1, nw)
        self.readSpectrumFile = ReadSpectrumFile(filename, w0, w1, nw)
        self.spectrum = self.readSpectrumFile.file_extension()
        if templateType == 'sn':
            self.wave, self.fluxes, self.nCols, self.ages, self.tType, self.splineInfo = self.spectrum
        elif templateType == 'gal':
            self.wave, self.flux = self.spectrum
            self.tType = self.filename
        else:
            print("INVALID ARGUMENT FOR TEMPLATE TYPE")

    def bin_template(self, ageIdx=None):
        if self.templateType == 'sn':
            if ageIdx is not None:
                return self._bin_sn_template(ageIdx)
            else:
                print("AGE INDEX ARGUMENT MISSING")
                return None
        elif self.templateType == 'gal':
            return self._bin_gal_template()
        else:
            print("INVALID ARGUMENT FOR TEMPLATE TYPE")
            return None

    def _bin_sn_template(self, ageIdx):
        wave, flux = self.readSpectrumFile.snid_template_undo_processing(self.wave, self.fluxes[ageIdx], self.splineInfo, ageIdx)
        binnedWave, binnedFlux, minIndex, maxIndex = self.preProcess.log_wavelength(wave, flux)
        binnedFluxNorm = normalise_spectrum(binnedFlux)
        binnedFluxNorm = zero_non_overlap_part(binnedFluxNorm, minIndex, maxIndex)

        return binnedWave, binnedFluxNorm, minIndex, maxIndex

    def _bin_gal_template(self):
        wave, flux = self.readSpectrumFile.two_col_input_spectrum(self.wave, self.flux, z=0)
        binnedWave, binnedFlux, minIndex, maxIndex = self.preProcess.log_wavelength(wave, flux)
        binnedFluxNorm = normalise_spectrum(binnedFlux)
        binnedFluxNorm = zero_non_overlap_part(binnedFluxNorm, minIndex, maxIndex)

        return binnedWave, binnedFluxNorm, minIndex, maxIndex


def create_sn_and_host_arrays(snTemplateDirectory, snTempFileList, galTemplateDirectory, galTempFileList, paramsFile):
    snTemplates = {}
    galTemplates = {}
    snList = TempList().temp_list(snTempFileList)
    galList = TempList().temp_list(galTempFileList)
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
    parameterFile = 'data_files/training_params.pickle'
    snTemplateDirectory = os.path.join(scriptDirectory, "../templates/snid_templates_Modjaz_BSNIP/")
    snTempFileList = snTemplateDirectory + 'templist.txt'
    galTemplateDirectory = os.path.join(scriptDirectory, "../templates/superfit_templates/gal/")
    galTempFileList = galTemplateDirectory + 'gal.list'
    saveFilename = 'models/sn_and_host_templates.npz'

    snTemplates, galTemplates = create_sn_and_host_arrays(snTemplateDirectory, snTempFileList, galTemplateDirectory, galTempFileList, parameterFile)

    np.savez_compressed(saveFilename, snTemplates=snTemplates, galTemplates=galTemplates)

    return saveFilename


if __name__ == "__main__":
    unCombinedTemplates = save_templates()
