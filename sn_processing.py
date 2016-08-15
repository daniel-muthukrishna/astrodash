#Pre-processing class

from preprocessing import *



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

        self.wave, self.flux = self.readInputSpectra.file_extension()


    def two_column_data(self, z):
        wave, flux = self.readInputSpectra.two_col_input_spectrum(self.wave, self.flux, z)
        binnedwave, binnedflux, minindex, maxindex = self.preProcess.log_wavelength(wave, flux)
        newflux = self.preProcess.continuum_removal(binnedwave, binnedflux, self.polyorder, minindex, maxindex)
        meanzero = self.preProcess.mean_zero(binnedwave, newflux, minindex, maxindex)
        apodized = self.preProcess.apodize(binnedwave, meanzero, minindex, maxindex)

        return binnedwave, apodized, minindex, maxindex

    def snid_template_data(self, ageidx, z):
        """lnw templates """
        
        wave, flux, ncols, ages, ttype = self.readInputSpectra.snid_template_spectra(ageidx, z)
        binnedwave, binnedflux, minindex, maxindex = self.preProcess.log_wavelength(wave, flux)
        
        return binnedwave, binnedflux, ncols, ages, ttype, minindex, maxindex




