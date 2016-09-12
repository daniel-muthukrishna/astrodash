#Pre-processing class

from preprocessing import *
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import medfilt, blackmanharris



class PreProcessing(object):
    """ Pre-processes spectra for cross correlation """
    
    def __init__(self, filename, w0, w1, nw):
        self.filename = filename
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.polyorder = 4
        self.readInputSpectra = ReadInputSpectra(self.filename, self.w0, self.w1)
        self.preProcess = PreProcessSpectrum(self.w0, self.w1, self.nw)

        self.spectrum = self.readInputSpectra.file_extension()


    def two_column_data(self, z):
        self.wave, self.flux = self.spectrum
        wave, flux = self.readInputSpectra.two_col_input_spectrum(self.wave, self.flux, z)
        binnedwave, binnedflux, minindex, maxindex = self.preProcess.log_wavelength(wave, flux)
        newflux, poly = self.preProcess.continuum_removal(binnedwave, binnedflux, self.polyorder, minindex, maxindex)
        meanzero = self.preProcess.mean_zero(binnedwave, newflux, minindex, maxindex)
        apodized = self.preProcess.apodize(binnedwave, meanzero, minindex, maxindex)

        medianFiltered = medfilt(apodized, kernel_size=3)

        # print wave
        # print binnedwave
        # print binnedflux
        # print len(binnedwave)
        # plt.figure('1')
        # plt.plot(wave,flux)
        # plt.figure('2')
        # plt.plot(binnedwave, binnedflux, label='binned')
        # plt.plot(binnedwave, newflux, label='continuumSubtract1')
        # plt.plot(binnedwave, poly, label='polyfit1')
        # #newflux2, poly2 = self.preProcess.continuum_removal(binnedwave, binnedflux, 6, minindex, maxindex)
        # #plt.plot(binnedwave, newflux2, label='continuumSubtract2')
        # #plt.plot(binnedwave, poly2, label='polyfit2')
        # plt.plot(binnedwave, apodized, label='taper')
        # plt.legend()
        # plt.figure('filtered')
        # plt.plot(medianFiltered)
        # plt.figure()
        # plt.plot(medfilt(apodized,kernel_size=5))
        # plt.show()

        return binnedwave, medianFiltered, minindex, maxindex

    def snid_template_data(self, ageidx, z):
        """lnw templates """
        wave, fluxes, ncols, ages, ttype = self.spectrum
        wave, flux = self.readInputSpectra.snid_template_spectra(wave, fluxes[ageidx], z)
        binnedwave, binnedflux, minindex, maxindex = self.preProcess.log_wavelength(wave, flux)
        medianFiltered = medfilt(binnedflux, kernel_size=3)

        return binnedwave, medianFiltered, ncols, ages, ttype, minindex, maxindex




