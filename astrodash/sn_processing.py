#Pre-processing class
import sys
from scipy.signal import medfilt
import numpy as np
from astrodash.preprocessing import ReadSpectrumFile, PreProcessSpectrum, ProcessingTools
from astrodash.array_tools import normalise_spectrum, zero_non_overlap_part


def limit_wavelength_range(wave, flux, minWave, maxWave):
    minIdx = (np.abs(wave-minWave)).argmin()
    maxIdx = (np.abs(wave-maxWave)).argmin()

    flux[:minIdx] = np.zeros(minIdx)
    flux[maxIdx:] = np.zeros(len(flux)-maxIdx)

    return flux


class PreProcessing(object):
    """ Pre-processes spectra before training """
    
    def __init__(self, filename, w0, w1, nw):
        self.filename = filename
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.numSplinePoints = 13
        self.processingTools = ProcessingTools()
        self.readSpectrumFile = ReadSpectrumFile(filename, w0, w1, nw)
        self.preProcess = PreProcessSpectrum(w0, w1, nw)

        self.spectrum = self.readSpectrumFile.file_extension()
        if len(self.spectrum) == 3:
            self.redshiftFromFile = True
        else:
            self.redshiftFromFile = False

    def two_column_data(self, z, smooth, minWave, maxWave):
        if self.redshiftFromFile is True:
            self.wave, self.flux, z = self.spectrum
        else:
            self.wave, self.flux = self.spectrum
        self.flux = normalise_spectrum(self.flux)
        self.flux = limit_wavelength_range(self.wave, self.flux, minWave, maxWave)
        self.wDensity = (self.w1 - self.w0)/self.nw  # Average wavelength spacing
        wavelengthDensity = (max(self.wave) - min(self.wave)) / len(self.wave)
        filterSize = int(self.wDensity / wavelengthDensity * smooth / 2) * 2 + 1
        preFiltered = medfilt(self.flux, kernel_size=filterSize)
        wave, deredshifted = self.readSpectrumFile.two_col_input_spectrum(self.wave, preFiltered, z)
        if len(wave) < 2:
            sys.exit("The redshifted spectrum of file: {0} is out of the classification range between {1} to {2} "
                     "Angstroms. Please remove this file from classification or reduce the redshift before re-running "
                     "the program.".format(self.filename, self.w0, self.w1))

        binnedwave, binnedflux, minIndex, maxIndex = self.preProcess.log_wavelength(wave, deredshifted)
        newflux, continuum = self.preProcess.continuum_removal(binnedwave, binnedflux, self.numSplinePoints, minIndex, maxIndex)
        meanzero = self.preProcess.mean_zero(newflux, minIndex, maxIndex)
        apodized = self.preProcess.apodize(meanzero, minIndex, maxIndex)


        #filterSize = smooth * 2 + 1
        medianFiltered = medfilt(apodized, kernel_size=1)#filterSize)
        fluxNorm = normalise_spectrum(medianFiltered)
        fluxNorm = zero_non_overlap_part(fluxNorm, minIndex, maxIndex, outerVal=0.5)


        # # PAPER PLOTS
        # import matplotlib.pyplot as plt
        #
        # plt.figure(num=1, figsize=(10, 6))
        # plt.plot(self.wave, self.flux, label='Raw', linewidth=1.3)
        # plt.plot(self.wave, preFiltered, label='Filtered', linewidth=1.3)
        # plt.ylim(-8, 8)
        # plt.xlabel('Wavelength ($\mathrm{\AA}$)', fontsize=14)
        # plt.ylabel('Relative Flux', fontsize=14)
        # plt.legend(fontsize=12, loc=1)
        # plt.xlim(2500, 9000)
        # plt.tick_params(labelsize=13)
        # plt.tight_layout()
        # plt.axhline(0, color='black', linewidth=0.5)
        # plt.savefig('/Users/danmuth/OneDrive/Documents/DASH/Paper/Figures/Filtering.pdf')
        #
        # plt.figure(num=2, figsize=(10, 6))
        # plt.plot(wave, deredshifted, label='De-redshifted', linewidth=1.3)
        # plt.plot(binnedwave, binnedflux, label='Log-wavelength binned', linewidth=1.3)
        # plt.xlabel('Wavelength ($\mathrm{\AA}$)', fontsize=14)
        # plt.ylabel('Relative Flux', fontsize=14)
        # plt.legend(fontsize=12, loc=1)
        # plt.xlim(2500, 9000)
        # plt.tick_params(labelsize=13)
        # plt.tight_layout()
        # plt.axhline(0, color='black', linewidth=0.5)
        # plt.savefig('/Users/danmuth/OneDrive/Documents/DASH/Paper/Figures/Deredshifting.pdf')
        #
        # plt.figure(num=3, figsize=(10, 6))
        # plt.plot(binnedwave, binnedflux, label='Log-wavelength binned', linewidth=1.3)
        # plt.plot(binnedwave, continuum, label='Continuum', linewidth=1.3)
        # plt.plot(binnedwave, newflux, label='Continuum divided', linewidth=1.3)
        # plt.xlabel('Wavelength ($\mathrm{\AA}$)', fontsize=14)
        # plt.ylabel('Relative Flux', fontsize=14)
        # plt.legend(fontsize=12, loc=1)
        # plt.xlim(2500, 9000)
        # plt.tick_params(labelsize=13)
        # plt.tight_layout()
        # plt.axhline(0, color='black', linewidth=0.5)
        # plt.savefig('/Users/danmuth/OneDrive/Documents/DASH/Paper/Figures/Continuum.pdf')
        #
        # plt.figure(num=4, figsize=(10, 6))
        # plt.plot(binnedwave, meanzero, label='Continuum divided', linewidth=1.3)
        # plt.plot(binnedwave, apodized, label='Apodized', linewidth=1.3)
        # # fluxNorm = (apodized - min(apodized)) / (max(apodized) - min(apodized))
        # plt.plot(binnedwave, fluxNorm, label='Normalised', linewidth=1.3)
        # plt.xlabel('Wavelength ($\mathrm{\AA}$)', fontsize=14)
        # plt.ylabel('Relative Flux', fontsize=14)
        # plt.legend(fontsize=12, loc=1)
        # plt.xlim(2500, 9000)
        # plt.tick_params(labelsize=13)
        # plt.tight_layout()
        # plt.axhline(0, color='black', linewidth=0.5)
        # plt.savefig('/Users/danmuth/OneDrive/Documents/DASH/Paper/Figures/Apodize.pdf')
        #
        # plt.show()

        return binnedwave, fluxNorm, minIndex, maxIndex, z



# fData = '/Users/danmuth/PycharmProjects/astrodash/templates/OzDES_data/ATEL_9570_Run25/DES16C2ma_C2_combined_160926_v10_b00.dat'
# preData = PreProcessing(fData, 3000, 10000, 1024)
# waveData,fluxData,minIData,maxIData,z = preData.two_column_data(0.24, 5, 3000, 10000)
