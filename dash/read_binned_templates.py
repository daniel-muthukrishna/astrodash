import numpy as np
import os
from dash.preprocessing import ProcessingTools, PreProcessSpectrum
from dash.array_tools import zero_non_overlap_part, normalise_spectrum
scriptDirectory = os.path.dirname(os.path.abspath(__file__))


def get_templates(snName, snAge, hostName, snTemplates, galTemplates, nw):
    snInfos = np.copy(snTemplates[snName][snAge]['snInfo'])
    snNames = np.copy(snTemplates[snName][snAge]['names'])
    if hostName != "No Host" and hostName != "":
        hostInfos = np.copy(galTemplates[hostName]['galInfo'])
        hostNames = np.copy(galTemplates[hostName]['names'])
    else:
        hostInfos = np.array([[np.zeros(nw), np.zeros(nw), 1, nw - 1]])
        hostNames = np.array(["No Host"])

    return snInfos, snNames, hostInfos, hostNames


def load_templates(templateFilename):
    loaded = np.load(os.path.join(scriptDirectory, templateFilename))
    snTemplates = loaded['snTemplates'][()]
    galTemplates = loaded['galTemplates'][()]

    return snTemplates, galTemplates


class ReadBinnedTemplates(object):
    def __init__(self, snInfo, galInfo, w0, w1, nw):
        self.snInfo = snInfo
        self.galInfo = galInfo
        self.processingTools = ProcessingTools()
        self.numSplinePoints = 13
        self.preProcess = PreProcessSpectrum(w0, w1, nw)

    def overlapped_spectra(self):
        snWave, snFlux, snMinIndex, snMaxIndex = self.snInfo
        galWave, galFlux, galMinIndex, galMaxIndex = self.galInfo

        minIndex = max(snMinIndex, galMinIndex)
        maxIndex = min(snMaxIndex, galMaxIndex)

        snFlux = zero_non_overlap_part(snFlux, minIndex, maxIndex)
        galFlux = zero_non_overlap_part(galFlux, minIndex, maxIndex)

        return snWave, snFlux, galWave, galFlux, minIndex, maxIndex

    def sn_plus_gal(self, snCoeff, galCoeff):
        snWave, snFlux, galWave, galFlux, minIndex, maxIndex = self.overlapped_spectra()

        combinedFlux = (snCoeff * snFlux) + (galCoeff * galFlux)

        return snWave, combinedFlux, minIndex, maxIndex

    def template_data(self, snCoeff, galCoeff, z):
        wave, flux, minIndex, maxIndex = self.sn_plus_gal(snCoeff, galCoeff)
        wave, flux = self.processingTools.redshift_spectrum(wave, flux, z)
        binnedWave, binnedFlux, minIndex, maxIndex = self.preProcess.log_wavelength(wave, flux)
        newFlux, continuum = self.preProcess.continuum_removal(binnedWave, binnedFlux, self.numSplinePoints, minIndex, maxIndex)
        meanZero = self.preProcess.mean_zero(binnedWave, newFlux, minIndex, maxIndex)
        apodized = self.preProcess.apodize(binnedWave, meanZero, minIndex, maxIndex)
        fluxNorm = normalise_spectrum(apodized)
        fluxNorm = zero_non_overlap_part(fluxNorm, minIndex, maxIndex)
        # Could  median filter here, but trying without it now

        return binnedWave, fluxNorm

if __name__ == "__main__":
    templateFilename1 = 'models/sn_and_host_templates.npz'
    snTemplates1, galTemplates1 = load_templates(templateFilename1)
    snInfoList = snTemplates1['Ia-norm']['-2 to 2']['snInfo']
    galInfoList = galTemplates1['S0']['galInfo']
    for i in range(len(snInfoList)):
        readBinnedTemplates = ReadBinnedTemplates(snInfoList[i], galInfoList[0], 2500, 10000, 1024)
        wave, flux = readBinnedTemplates.template_data(snCoeff=0.5, galCoeff=0.5, z=0)
        print(i)
