from dash.preprocessing import ReadSpectrumFile, ProcessingTools, PreProcessSpectrum
from dash.array_tools import zero_non_overlap_part, normalise_spectrum
import matplotlib.pyplot as plt
import numpy as np


class CombineSnAndHost(object):
    def __init__(self, snFile, galFile, w0, w1, nw):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.numSplinePoints = 13
        self.processingTools = ProcessingTools()
        self.snReadSpectrumFile = ReadSpectrumFile(snFile, w0, w1, nw)
        self.galReadSpectrumFile = ReadSpectrumFile(galFile, w0, w1, nw)
        self.snSpectrum = self.snReadSpectrumFile.file_extension()
        self.snWave, self.snFluxes, self.ncols, self.ages, self.ttype, self.splineInfo = self.snSpectrum
        self.galSpectrum = self.galReadSpectrumFile.file_extension()
        self.preProcess = PreProcessSpectrum(w0, w1, nw)

    def snid_sn_template_data(self, ageIdx):
        # Undo continuum in the following step in preprocessing.py
        wave, flux = self.snReadSpectrumFile.snid_template_undo_processing(self.snWave, self.snFluxes[ageIdx], self.splineInfo, ageIdx)

        binnedWave, binnedFlux, minIndex, maxIndex = self.preProcess.log_wavelength(wave, flux)
        binnedFluxNorm = normalise_spectrum(binnedFlux)

        return binnedWave, binnedFluxNorm, minIndex, maxIndex

    def gal_template_data(self):
        wave, flux = self.galSpectrum

        # Limit bounds from w0 to w1 and normalise flux
        wave, flux = self.galReadSpectrumFile.two_col_input_spectrum(wave, flux, z=0)
        binnedWave, binnedFlux, minIndex, maxIndex = self.preProcess.log_wavelength(wave, flux)
        binnedFluxNorm = normalise_spectrum(binnedFlux)

        return binnedWave, binnedFluxNorm, minIndex, maxIndex

    def overlapped_spectra(self, snAgeIdx):
        snWave, snFlux, snMinIndex, snMaxIndex = self.snid_sn_template_data(snAgeIdx)
        galWave, galFlux, galMinIndex, galMaxIndex = self.gal_template_data()

        minIndex = max(snMinIndex, galMinIndex)
        maxIndex = min(snMaxIndex, galMaxIndex)

        snFlux = zero_non_overlap_part(snFlux, minIndex, maxIndex)
        galFlux = zero_non_overlap_part(galFlux, minIndex, maxIndex)

        return snWave, snFlux, galWave, galFlux, minIndex, maxIndex

    def sn_plus_gal(self, snCoeff, galCoeff, snAgeIdx):
        snWave, snFlux, galWave, galFlux, minIndex, maxIndex = self.overlapped_spectra(snAgeIdx)

        combinedFlux = (snCoeff * snFlux) + (galCoeff * galFlux)

        return snWave, combinedFlux, minIndex, maxIndex

    def training_template_data(self, snAgeIdx, snCoeff, galCoeff, z):
        wave, flux, minIndex, maxIndex = self.sn_plus_gal(snCoeff, galCoeff, snAgeIdx)
        wave, flux = self.processingTools.redshift_spectrum(wave, flux, z)
        binnedWave, binnedFlux, minIndex, maxIndex = self.preProcess.log_wavelength(wave, flux)
        newFlux, continuum = self.preProcess.continuum_removal(binnedWave, binnedFlux, self.numSplinePoints, minIndex, maxIndex)
        meanZero = self.preProcess.mean_zero(binnedWave, newFlux, minIndex, maxIndex)
        apodized = self.preProcess.apodize(binnedWave, meanZero, minIndex, maxIndex)
        # fluxNorm = self._normalise_spectrum(apodized) # This happens in create_arrays anyway
        # Could  median filter here, but trying without it now

        return binnedWave, apodized, minIndex, maxIndex, self.ncols, self.ages, self.ttype



if __name__ == '__main__':
    fSN = '/Users/danmuth/PycharmProjects/DASH/templates/snid_templates_Modjaz_BSNIP/sn2001br.lnw'
    fGal = '/Users/danmuth/PycharmProjects/DASH/templates/superfit_templates/gal/Sa'
    combine = CombineSnAndHost(fSN, fGal, 2500, 10000, 1024)

    f = plt.figure()
    xSN, ySN, minSN, maxSN = combine.snid_sn_template_data(ageIdx=0)
    xGal, yGal, minGal, maxGal = combine.gal_template_data()
    plt.plot(xSN, ySN, 'b')
    plt.plot(xGal, yGal, 'r')

    f2 = plt.figure()
    xSN, ySN, xGal, yGal, minI, maxI = combine.overlapped_spectra(0)
    xCombined, yCombined, minI, maxI = combine.sn_plus_gal(0.5, 0.5, 0)
    xTemp, ytemp, minI, maxI, NCOL, AGE, TTYPE = combine.training_template_data(0, 0.5, 0.5, 0)
    plt.plot(xSN, ySN, 'b')
    plt.plot(xGal, yGal, 'r')
    plt.plot(xCombined, yCombined, 'g')
    plt.plot(xTemp, ytemp, 'c')


    # galNames = ['E', 'S0', 'Sa', 'Sb', 'Sc', 'SB1', 'SB2', 'SB3', 'SB4', 'SB5', 'SB6',]
    # fGals = ['/Users/danmuth/PycharmProjects/DASH/templates/superfit_templates/gal/E',
    #         '/Users/danmuth/PycharmProjects/DASH/templates/superfit_templates/gal/S0',
    #         '/Users/danmuth/PycharmProjects/DASH/templates/superfit_templates/gal/Sa',
    #         '/Users/danmuth/PycharmProjects/DASH/templates/superfit_templates/gal/Sb',
    #         '/Users/danmuth/PycharmProjects/DASH/templates/superfit_templates/gal/Sc',
    #         '/Users/danmuth/PycharmProjects/DASH/templates/superfit_templates/gal/SB1',
    #         '/Users/danmuth/PycharmProjects/DASH/templates/superfit_templates/gal/SB2',
    #         '/Users/danmuth/PycharmProjects/DASH/templates/superfit_templates/gal/SB3',
    #         '/Users/danmuth/PycharmProjects/DASH/templates/superfit_templates/gal/SB4',
    #         '/Users/danmuth/PycharmProjects/DASH/templates/superfit_templates/gal/SB5',
    #         '/Users/danmuth/PycharmProjects/DASH/templates/superfit_templates/gal/SB6',]
    #
    # for i in range(len(fGals)):
    #     f3 = plt.figure()
    #     combine = CombineSnAndHost(fSN, fGals[i], 2500, 10000, 1024)
    #     xGal, yGal, minGal, maxGal = combine.gal_template_data()
    #     plt.plot(xGal, yGal)
    #     plt.savefig("%s.png" % galNames[i])

    plt.show()




