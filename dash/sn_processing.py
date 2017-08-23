#Pre-processing class

from scipy.signal import medfilt
import numpy as np
from dash.preprocessing import ReadSpectrumFile, PreProcessSpectrum, ProcessingTools


def limit_wavelength_range(wave, flux, minWave, maxWave):
    minIdx = (np.abs(wave-minWave)).argmin()
    maxIdx = (np.abs(wave-maxWave)).argmin()

    flux[:minIdx] = np.zeros(minIdx)
    flux[maxIdx:] = np.zeros(len(flux)-maxIdx)

    return flux


class PreProcessing(object):
    """ Pre-processes spectra for cross correlation """
    
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

    def galaxy_template(self, z):
        self.wave, self.flux = self.spectrum
        wave, flux = self.readSpectrumFile.two_col_input_spectrum(self.wave, self.flux, z)
        binnedwave, binnedflux, minindex, maxindex = self.preProcess.log_wavelength(wave, flux)
        newflux, continuum = self.preProcess.continuum_removal(binnedwave, binnedflux, self.numSplinePoints, minindex, maxindex)
        meanzero = self.preProcess.mean_zero(binnedwave, newflux, minindex, maxindex)
        apodized = self.preProcess.apodize(binnedwave, meanzero, minindex, maxindex)

        plt.figure('Galaxy_SB1')
        plt.plot(wave,flux, label='original')
        plt.plot(binnedwave,binnedflux, label='binned')
        plt.plot(binnedwave, continuum, label='continuum')
        plt.plot(binnedwave,newflux, label='continuumSubtracted')
        plt.plot(binnedwave, meanzero, label='meanzero')
        plt.plot(binnedwave,apodized, label='apodized')
        plt.legend()
        #plt.show()

        return binnedwave, apodized, minindex, maxindex


    def two_column_data(self, z, smooth, minWave, maxWave):
        self.wave, self.flux = self.spectrum
        self.flux = limit_wavelength_range(self.wave, self.flux, minWave, maxWave)
        self.wDensity = (self.w1 - self.w0)/self.nw  # Average wavelength spacing
        wavelengthDensity = (max(self.wave) - min(self.wave)) / len(self.wave)

        filterSize = int(self.wDensity / wavelengthDensity * smooth / 2) * 2 + 1
        preFiltered = medfilt(self.flux, kernel_size=filterSize)
        wave, flux = self.readSpectrumFile.two_col_input_spectrum(self.wave, preFiltered, z)
        binnedwave, binnedflux, minindex, maxindex = self.preProcess.log_wavelength(wave, flux)
        newflux, continuum = self.preProcess.continuum_removal(binnedwave, binnedflux, self.numSplinePoints, minindex, maxindex)
        meanzero = self.preProcess.mean_zero(binnedwave, newflux, minindex, maxindex)
        apodized = self.preProcess.apodize(binnedwave, meanzero, minindex, maxindex)


        #filterSize = smooth * 2 + 1
        medianFiltered = medfilt(apodized, kernel_size=1)#filterSize)


        # # PAPER PLOTS
        # import matplotlib.pyplot as plt
        #
        # plt.figure(num=1, figsize=(10, 6))
        # plt.plot(self.wave, self.flux, label='Raw', linewidth=1.3)
        # plt.plot(self.wave, preFiltered, label='Filtered', linewidth=1.3)
        # plt.ylim(-8, 8)
        # plt.xlabel('Wavelength ($\mathrm{\AA}$)', fontsize=13)
        # plt.ylabel('Relative Flux', fontsize=13)
        # plt.legend(fontsize=11, loc=1)
        # plt.xlim(2500, 9000)
        # plt.tight_layout()
        # plt.axhline(0, color='black', linewidth=0.5)
        # plt.savefig('/Users/danmuth/OneDrive/Documents/DASH/Paper/Figures/Filtering.png')
        #
        # plt.figure(num=2, figsize=(10, 6))
        # plt.plot(self.wave, preFiltered, label='Filtered', linewidth=1.3)
        # plt.plot(binnedwave, binnedflux, label='De-redshifted and log-wavelength binned', linewidth=1.3)
        # plt.xlabel('Wavelength ($\mathrm{\AA}$)', fontsize=13)
        # plt.ylabel('Relative Flux', fontsize=13)
        # plt.legend(fontsize=11, loc=1)
        # plt.xlim(2500, 9000)
        # plt.tight_layout()
        # plt.axhline(0, color='black', linewidth=0.5)
        # plt.savefig('/Users/danmuth/OneDrive/Documents/DASH/Paper/Figures/Deredshifting.png')
        #
        # plt.figure(num=3, figsize=(10, 6))
        # plt.plot(binnedwave, binnedflux, label='Log-wavelength binned', linewidth=1.3)
        # plt.plot(binnedwave, continuum, label='Continuum', linewidth=1.3)
        # plt.plot(binnedwave, newflux, label='Continuum subtracted', linewidth=1.3)
        # plt.xlabel('Wavelength ($\mathrm{\AA}$)', fontsize=13)
        # plt.ylabel('Relative Flux', fontsize=13)
        # plt.legend(fontsize=11, loc=1)
        # plt.xlim(2500, 9000)
        # plt.tight_layout()
        # plt.axhline(0, color='black', linewidth=0.5)
        # plt.savefig('/Users/danmuth/OneDrive/Documents/DASH/Paper/Figures/Continuum.png')
        #
        # plt.figure(num=4, figsize=(10, 6))
        # plt.plot(binnedwave, newflux, label='Continuum subtracted', linewidth=1.3)
        # plt.plot(binnedwave, apodized, label='Apodized', linewidth=1.3)
        # fluxNorm = (apodized - min(apodized)) / (max(apodized) - min(apodized))
        # plt.plot(binnedwave, fluxNorm, label='Normalised', linewidth=1.3)
        # plt.xlabel('Wavelength ($\mathrm{\AA}$)', fontsize=13)
        # plt.ylabel('Relative Flux', fontsize=13)
        # plt.legend(fontsize=11, loc=1)
        # plt.xlim(2500, 9000)
        # plt.tight_layout()
        # plt.axhline(0, color='black', linewidth=0.5)
        # plt.savefig('/Users/danmuth/OneDrive/Documents/DASH/Paper/Figures/Apodize.png')
        #
        # plt.show()

        return binnedwave, medianFiltered, minindex, maxindex

    def snid_template_data(self, ageIdx, z):
        """lnw templates """
        wave, fluxes, ncols, ages, ttype, splineInfo = self.spectrum
        wave, flux = self.processingTools.redshift_spectrum(wave, fluxes[ageIdx], z)
        binnedwave, binnedflux, minindex, maxindex = self.preProcess.log_wavelength(wave, flux)
        medianFiltered = medfilt(binnedflux, kernel_size=1)

        return binnedwave, medianFiltered, ncols, ages, ttype, minindex, maxindex



# fData = '/Users/danmuth/PycharmProjects/DASH/templates/OzDES_data/ATEL_9570_Run25/DES16C2ma_C2_combined_160926_v10_b00.dat'
# preData = PreProcessing(fData, 2500, 10000, 1024)
# waveData,fluxData,minIData,maxIData = preData.two_column_data(0.24, 5, 2500, 10000)