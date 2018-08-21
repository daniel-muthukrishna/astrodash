import os
from scipy.signal import medfilt
from astrodash.preprocessing import ReadSpectrumFile, ProcessingTools, PreProcessSpectrum
from astrodash.array_tools import zero_non_overlap_part, normalise_spectrum


class CombineSnAndHost(object):
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
        flux = zero_non_overlap_part(flux, minIndex, maxIndex, outerVal=0)
        binnedWave, binnedFlux, minIndex, maxIndex = self.preProcess.log_wavelength(wave[minIndex:maxIndex+1], flux[minIndex:maxIndex+1])
        newFlux, continuum = self.preProcess.continuum_removal(binnedWave, binnedFlux, self.numSplinePoints, minIndex, maxIndex)
        meanZero = self.preProcess.mean_zero(newFlux, minIndex, maxIndex)
        apodized = self.preProcess.apodize(meanZero, minIndex, maxIndex)
        fluxNorm = normalise_spectrum(apodized)
        fluxNorm = zero_non_overlap_part(fluxNorm, minIndex, maxIndex, outerVal=0.5)
        # Could  median filter here, but trying without it now

        return binnedWave, fluxNorm, (minIndex, maxIndex)


class BinTemplate(object):
    def __init__(self, filename, templateType, w0, w1, nw):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.numSplinePoints = 13
        self.filename = os.path.basename(filename)
        self.templateType = templateType
        self.preProcess = PreProcessSpectrum(w0, w1, nw)
        self.readSpectrumFile = ReadSpectrumFile(filename, w0, w1, nw)
        self.spectrum = self.readSpectrumFile.file_extension(template=True)
        if templateType == 'sn':
            if len(self.spectrum) == 6:
                self.snidTemplate = True
                self.wave, self.fluxes, self.nCols, self.ages, self.tType, self.splineInfo = self.spectrum
            else:
                self.snidTemplate = False
                self.wave, self.flux, self.nCols, self.ages, self.tType = self.spectrum

        elif templateType == 'gal':
            self.wave, self.flux = self.spectrum
            self.tType = self.filename
        else:
            print("INVALID ARGUMENT FOR TEMPLATE TYPE")

    def bin_template(self, ageIdx=None):
        if self.templateType == 'sn':
            if ageIdx is None and self.snidTemplate:
                print("AGE INDEX ARGUMENT MISSING")
                return None
            else:
                return self._bin_sn_template(ageIdx)

        elif self.templateType == 'gal':
            return self._bin_gal_template()
        else:
            print("INVALID ARGUMENT FOR TEMPLATE TYPE")
            return None

    def _bin_sn_template(self, ageIdx):
        # Undo continuum in the following step in preprocessing.py
        if self.snidTemplate:
            wave, flux = self.wave, self.fluxes[ageIdx]  # self.snReadSpectrumFile.snid_template_undo_processing(self.snWave, self.snFluxes[ageIdx], self.splineInfo, ageIdx)
            binnedWave, binnedFlux, minIndex, maxIndex = self.preProcess.log_wavelength(wave, flux)
            binnedFluxNorm = normalise_spectrum(binnedFlux)
            binnedFluxNorm = zero_non_overlap_part(binnedFluxNorm, minIndex, maxIndex, outerVal=0.5)
        else:
            wave, flux = self.wave, medfilt(self.flux, kernel_size=3)
            flux = normalise_spectrum(flux)
            binnedWave, binnedFlux, minIndex, maxIndex = self.preProcess.log_wavelength(wave, flux)
            contRemovedFlux, continuum = self.preProcess.continuum_removal(binnedWave, binnedFlux, self.numSplinePoints, minIndex, maxIndex)
            meanzero = self.preProcess.mean_zero(contRemovedFlux, minIndex, maxIndex)
            apodized = self.preProcess.apodize(meanzero, minIndex, maxIndex)
            binnedFluxNorm = normalise_spectrum(apodized)
            binnedFluxNorm = zero_non_overlap_part(binnedFluxNorm, minIndex, maxIndex, outerVal=0.5)

        return binnedWave, binnedFluxNorm, minIndex, maxIndex

    def _bin_gal_template(self):
        wave, flux = self.readSpectrumFile.two_col_input_spectrum(self.wave, self.flux, z=0)
        # flux = normalise_spectrum(flux)
        binnedWave, binnedFlux, minIndex, maxIndex = self.preProcess.log_wavelength(wave, flux)
        contRemovedFlux, continuum = self.preProcess.continuum_removal(binnedWave, binnedFlux, self.numSplinePoints, minIndex, maxIndex)
        newFlux = contRemovedFlux * continuum  # Spectral features weighted by the continuum
        fluxNorm = normalise_spectrum(newFlux)
        fluxNorm = zero_non_overlap_part(fluxNorm, minIndex, maxIndex, outerVal=0.5)

        return binnedWave, fluxNorm, minIndex, maxIndex


def training_template_data(snAgeIdx, snCoeff, galCoeff, z, snFile, galFile, w0, w1, nw):
    snBinTemplate = BinTemplate(filename=snFile, templateType='sn', w0=w0, w1=w1, nw=nw)
    snInfo = snBinTemplate.bin_template(snAgeIdx)

    if galFile is None:  # No host combining
        binnedWave, fluxNorm, minIndex, maxIndex = snInfo
    else:
        galBinTemplate = BinTemplate(filename=galFile, templateType='gal', w0=w0, w1=w1, nw=nw)
        galInfo = galBinTemplate.bin_template()
        combineSnAndHost = CombineSnAndHost(snInfo, galInfo, w0, w1, nw)
        binnedWave, fluxNorm, (minIndex, maxIndex) = combineSnAndHost.template_data(snCoeff, galCoeff, z)

    nCols, ages, tType = snBinTemplate.nCols, snBinTemplate.ages, snBinTemplate.tType

    return binnedWave, fluxNorm, minIndex, maxIndex, nCols, ages, tType
