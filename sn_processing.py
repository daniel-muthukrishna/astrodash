#Pre-processing class

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from preprocessing import *



class PreProcessing(object):
    """ Pre-processes spectra for cross correlation """
    
    def __init__(self):
        pass#self.filename = filename

    def processed_data(self, filename, w0, w1, nw):
        polyorder = 4
        
        wave, flux = input_spectra(filename, w0, w1)
        binnedwave, binnedflux, minindex, maxindex = log_wavelength(wave, flux, w0, w1, nw)
        polyx, polyy = poly_fit(binnedwave, binnedflux, polyorder, minindex, maxindex)
        newflux = continuum_removal(binnedwave, binnedflux, polyorder, minindex, maxindex)
        meanzero = mean_zero(binnedwave, newflux, minindex, maxindex)
        apodized = apodize(binnedwave, meanzero, minindex, maxindex)

        return binnedwave, apodized, minindex, maxindex

    def templates(self, filename, ageidx, w0, w1, nw):
        """lnw templates """
        
        wave, flux, ncols, ages, ttype = template_spectra(filename, ageidx)
        binnedwave, binnedflux, minindex, maxindex = log_wavelength(wave, flux, w0, w1, nw)
        
        return binnedwave, binnedflux, ncols, ages, ttype, minindex, maxindex

    def galaxy_template(self, filename, w0, w1, nw):
        """superfit galaxy templates"""
        polyorder = 4
        
        wave, flux = input_spectra(filename)
        binnedwave, binnedflux, minindex, maxindex = log_wavelength(wave, flux, w0, w1, nw)
        polyx, polyy = poly_fit(binnedwave, binnedflux, polyorder, minindex, maxindex)
        newflux = continuum_removal(binnedwave, binnedflux, polyorder, minindex, maxindex)
        meanzero = mean_zero(binnedwave, newflux, minindex, maxindex)
        apodized = apodize(binnedwave, meanzero, minindex, maxindex)

        return binnedwave, apodized






##filelocation = 'C:\Users\Daniel\OneDrive\Documents\Thesis Project\superfit\sne\Ia\\'
##snfilename ='sn2002bo.m01.dat' #sn2003jo.dat
##templatefilename = 'sn1999ee.m08.dat'
##
##inputpre = PreProcessing()
##temppre = PreProcessing()
##
##
##wd,fd = inputpre.processed_data(filelocation+snfilename)
##wt,ft = inputpre.processed_data(filelocation+templatefilename)
##
##plt.plot(wd,fd)
##plt.plot(wt,ft)
##
##plt.show()

