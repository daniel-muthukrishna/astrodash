from dash.preprocessing import ReadSpectrumFile, ProcessingTools, PreProcessSpectrum


class CombineSnAndHost(object):
    def __init__(self, snFile, galFile, w0, w1, nw):
        self.snSpectrum = ReadSpectrumFile(snFile, w0, w1, nw).file_extension()
        self.galSpectrum = ReadSpectrumFile(galFile, w0, w1, nw).file_extension()

    def sn_plus_gal(self):
        self.snIndex