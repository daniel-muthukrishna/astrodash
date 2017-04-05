from dash.preprocessing import ReadSpectrumFile, ProcessingTools, PreProcessSpectrum
import matplotlib.pyplot as plt


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
        self.galSpectrum = self.galReadSpectrumFile.file_extension()

    def snid_sn_template_data(self, ageIdx):
        wave, fluxes, ncols, ages, ttype, splineInfo = self.snSpectrum
        # Undo Binning in the following step in preprocessing.py
        wave, flux = self.snReadSpectrumFile.snid_template_undo_processing(wave, fluxes[ageIdx], splineInfo)

        # Limit bounds from w0 to w1 and normalise flux
        wave, flux = self.snReadSpectrumFile.two_col_input_spectrum(wave, flux, z=0)

        return wave, flux

    def gal_template_data(self):
        wave, flux = self.galSpectrum

        # Limit bounds from w0 to w1 and normalise flux
        wave, flux = self.galReadSpectrumFile.two_col_input_spectrum(wave, flux, z=0)

        return wave, flux

    def overlapped_spectra(self, snAgeIdx):
        snWave, snFlux = self.snid_sn_template_data(snAgeIdx)
        galWave, galFlux = self.gal_template_data()
        snIndexes = self.processingTools.min_max_index(snFlux)
        galIndexes = self.processingTools.min_max_index(galFlux)

        minIndex = min(snIndexes[0], galIndexes[0])
        maxIndex = max(snIndexes[1], galIndexes[1])

        snWave = snWave[minIndex:maxIndex]
        snFlux = snFlux[minIndex:maxIndex]
        galWave = galWave[minIndex:maxIndex]
        galFlux = galFlux[minIndex:maxIndex]

        return (snWave, snFlux), (galWave, galFlux)


    def sn_plus_gal(self, snCoeff, galCoeff):
        snSpectrumOverlapped, galSpectrumOverlapped = self._overlapped_spectra()

        combinedSpectrum = 0




if __name__ == '__main__':
    fSN = '/Users/danmuth/PycharmProjects/DASH/templates/snid_templates_Modjaz_BSNIP/sn2001br.lnw'
    gGal = '/Users/danmuth/PycharmProjects/DASH/templates/superfit_templates/gal/Sa'
    combine = CombineSnAndHost(fSN, gGal, 2500, 10000, 1024)

    f = plt.figure()
    plt.plot(combine.snid_sn_template_data(ageIdx=0))
    plt.plot(combine.gal_template_data())

    f2 = plt.figure()

    plt.plot(combine.overlapped_spectra(0)[0])
    plt.plot(combine.overlapped_spectra(0)[1])
    plt.show()



