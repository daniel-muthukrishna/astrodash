import numpy as np
import pickle
import os
from dash.preprocessing import ReadSpectrumFile, ProcessingTools, PreProcessSpectrum
from dash.create_arrays import TempList


def normalise_spectrum(flux):
    fluxNorm = (flux - min(flux)) / (max(flux) - min(flux))

    return fluxNorm


class BinTemplate(object):
    def __init__(self, filename, templateType, w0, w1, nw):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.templateType = templateType
        self.preProcess = PreProcessSpectrum(w0, w1, nw)
        self.readSpectrumFile = ReadSpectrumFile(filename, w0, w1, nw)
        self.spectrum = self.readSpectrumFile.file_extension()
        if templateType == 'sn':
            self.wave, self.fluxes, self.nCols, self.ages, self.tType, self.splineInfo = self.spectrum
        elif templateType == 'gal':
            self.wave, self.flux = self.spectrum
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

        return binnedWave, binnedFluxNorm, minIndex, maxIndex

    def _bin_gal_template(self):
        wave, flux = self.readSpectrumFile.two_col_input_spectrum(self.wave, self.flux, z=0)
        binnedWave, binnedFlux, minIndex, maxIndex = self.preProcess.log_wavelength(wave, flux)
        binnedFluxNorm = normalise_spectrum(binnedFlux)

        return binnedWave, binnedFluxNorm, minIndex, maxIndex


def create_sn_and_host_arrays(snTemplateDirectory, snTempFileList, galTemplateDirectory, galTempFileList, paramsFile):
    snInfoList = []
    galInfoList = []
    snList = TempList().temp_list(snTempFileList)
    galList = TempList().temp_list(galTempFileList)
    with open(paramsFile, 'rb') as f:
        pars = pickle.load(f)
    w0, w1, nw = pars['w0'], pars['w1'], pars['nw']

    for snFile in snList:
        snBinTemplate = BinTemplate(snTemplateDirectory + snFile, 'sn', w0, w1, nw)
        nAges = snBinTemplate.nCols
        for ageIdx in range(nAges):
            snInfo = snBinTemplate.bin_template(ageIdx)
            snInfoList.append(snInfo)
            print("Reading {} {} out of {}".format(snFile, ageIdx, nAges))

    for galFile in galList:
        galBinTemplate = BinTemplate(galTemplateDirectory + galFile, 'gal', w0, w1, nw)
        galInfo = galBinTemplate.bin_template()
        galInfoList.append(galInfo)
        print("Reading {}".format(galFile))

    return np.array(snInfoList), np.array(galInfoList)


def save_templates():
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    parameterFile = 'data_files/training_params.pickle'
    snTemplateDirectory = os.path.join(scriptDirectory, "../templates/snid_templates_Modjaz_BSNIP/")
    snTempFileList = snTemplateDirectory + 'templist.txt'
    galTemplateDirectory = os.path.join(scriptDirectory, "../templates/superfit_templates/gal/")
    galTempFileList = galTemplateDirectory + 'gal.list'
    saveFilename = 'sn_and_host_templates.npz'

    snInfoList, galInfoList = create_sn_and_host_arrays(snTemplateDirectory, snTempFileList, galTemplateDirectory, galTempFileList, parameterFile)

    np.savez_compressed(saveFilename, snInfoList=snInfoList, galInfoList=galInfoList)

    return saveFilename


if __name__ == "__main__":
    unCombinedTemplates = save_templates()
