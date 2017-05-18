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


        # from scipy.interpolate import interp1d
        #
        # plt.plot(self.flux)
        #
        # spline = interp1d(binnedwave[minindex:maxindex], binnedflux[minindex:maxindex], kind='cubic')
        # waveSpline = np.linspace(binnedwave[minindex],binnedwave[maxindex-1],num=self.numSplinePoints)
        # print(spline)
        # print('###')
        # print(spline(binnedwave[minindex:maxindex]))
        # plt.figure('1')
        # plt.plot(waveSpline, spline(waveSpline), '--', label='spline')
        #
        # print(wave)
        # print(binnedwave)
        # print(binnedflux)
        # print(len(binnedwave))
        # plt.plot(wave,flux)
        # plt.figure('2')
        # plt.plot(binnedwave, binnedflux, label='binned')
        # plt.plot(binnedwave, newflux, label='continuumSubtract1')
        # plt.plot(binnedwave, continuum, label='polyfit1')
        # print(len(binnedwave))
        # print((min(binnedwave), max(binnedwave)))
        # print(len(newflux))
        #
        # #newflux2, poly2 = self.preProcess.continuum_removal(binnedwave, binnedflux, 6, minindex, maxindex)
        # #plt.plot(binnedwave, newflux2, label='continuumSubtract2')
        # #plt.plot(binnedwave, poly2, label='polyfit2')
        # plt.plot(binnedwave, apodized, label='taper')
        # plt.legend()
        # plt.figure('filtered')
        # plt.plot(medianFiltered)
        # plt.figure('3')
        # plt.plot(medfilt(apodized,kernel_size=3))
        # plt.show()

        return binnedwave, medianFiltered, minindex, maxindex

    def snid_template_data(self, ageIdx, z):
        """lnw templates """
        wave, fluxes, ncols, ages, ttype, splineInfo = self.spectrum
        wave, flux = self.processingTools.redshift_spectrum(wave, fluxes[ageIdx], z)
        binnedwave, binnedflux, minindex, maxindex = self.preProcess.log_wavelength(wave, flux)
        medianFiltered = medfilt(binnedflux, kernel_size=1)

        return binnedwave, medianFiltered, ncols, ages, ttype, minindex, maxindex

# if __name__ == '__main__':
#     # fData = '/home/dan/Desktop/SNClassifying_Pre-alpha/templates/superfit_templates/sne/Ia/sn2002bo.m01.dat'
#     # preData = PreProcessing(fGal, 2500, 10000, 1024)
#     # waveData,fluxData,minIData,maxIData = preData.two_column_data(0, 0)
#
#     fGal = '/home/dan/Desktop/SNClassifying_Pre-alpha/templates/galaxy_templates/GalaxyTemplates/Sa'
#     preGal = PreProcessing(fGal, 2500, 10000, 1024)
#     waveGal,fluxGal,minIGal,maxIGal = preGal.galaxy_template(0)
#
#     fSN = '/home/dan/Desktop/SNClassifying_Pre-alpha/templates/snid_templates_Modjaz_BSNIP/sn2001br.lnw'
#     preSN = PreProcessing(fSN, 2500, 10000, 1024)
#     waveSN, fluxSN, ncols, ages, ttype, minISN, maxISN = preSN.snid_template_data(0, 0)
#
#     plt.figure('SN and Galaxy Template')
#     minI = max(minIGal, minISN)
#     maxI = min(maxIGal, maxISN)
#     plt.axvline(x=waveSN[minI], color='k', linestyle='--')
#     plt.axvline(x=waveSN[maxI], color='k', linestyle='--')
#
#
#
#     plt.plot(waveGal,fluxGal, label='galaxy')
#     plt.plot(waveSN,fluxSN, label='SN')
#
#     waveSN = waveSN[minI:maxI]
#     fluxSN = fluxSN[minI:maxI]
#     fluxGal = fluxGal[minI:maxI]
#
#     plt.plot(waveSN,fluxGal+fluxSN, label='Added-50%')
#     plt.legend()
#     plt.show()
