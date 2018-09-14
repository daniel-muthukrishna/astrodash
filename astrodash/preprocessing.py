import os
import sys
import numpy as np
import astropy.io.fits as afits
from scipy.interpolate import interp1d, UnivariateSpline
from astrodash.array_tools import normalise_spectrum, zero_non_overlap_part
from astrodash.read_from_catalog import catalogDict
try:
    import pandas as pd
    USE_PANDAS = True
except ImportError:
    print("Pandas module not installed. DASH will use numpy to load spectral files instead. "
          "This can be up to 10x slower.")
    USE_PANDAS = False


class ProcessingTools(object):
    def redshift_spectrum(self, wave, flux, z):
        wave_new = wave * (z + 1)

        return wave_new, flux

    def deredshift_spectrum(self, wave, flux, z):
        wave_new = wave / (z + 1)

        return wave_new, flux

    def min_max_index(self, flux, outerVal=0):
        """ 
        :param flux: 
        :param outerVal: is the scalar value in all entries before the minimum and after the maximum index
        :return: 
        """
        nonZeros = np.where(flux != outerVal)[0]
        if nonZeros.size:
            minIndex, maxIndex = min(nonZeros), max(nonZeros)
        else:
            minIndex, maxIndex = len(flux), len(flux)

        return minIndex, maxIndex


class ReadSpectrumFile(object):
    def __init__(self, filename, w0, w1, nw):
        self.filename = filename
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.processingTools = ProcessingTools()

    def read_fits_file(self):
        # filename = unicode(self.filename.toUtf8(), encoding="UTF-8")
        try:
            if 'specutils' not in sys.modules:
                from specutils.io import read_fits

            spectrum = read_fits.read_fits_spectrum1d(self.filename)
            if len(spectrum) > 1:
                spectrum = spectrum[0]
            try:
                wave = np.array(spectrum.wavelength)
                flux = np.array(spectrum.flux)
            except AttributeError:
                wave = np.array(spectrum.dispersion)
                flux = np.array(spectrum.flux)
                print("No wavelength attribute in FITS File. Using 'dispersion' attribute instead")
        except Exception as e:
            hdulist = afits.open(self.filename)
            flux = hdulist[0].data
            wave_start = hdulist[0].header['CRVAL1']
            if 'CD1_1' in hdulist[0].header:
                wave_step = hdulist[0].header['CD1_1']
            else:
                wave_step = hdulist[0].header['CDELT1']
            wave_num = flux.shape[0]
            wave = np.linspace(wave_start, wave_start + wave_step * wave_num, num=wave_num)

        flux[np.isnan(flux)] = 0  # convert nan's to zeros

        return wave, flux

    def read_dat_file(self):
        try:
            if USE_PANDAS is True:
                data = pd.read_csv(self.filename, header=None, delim_whitespace=True).values
            else:
                data = np.loadtxt(self.filename)
            wave = data[:, 0]
            flux = data[:, 1]
        except:
            print("COULDN'T USE LOADTXT FOR FILE: {0}\n READ LINE BY LINE INSTEAD.".format(self.filename))
            wave = []
            flux = []
            with open(self.filename, 'r') as FileObj:
                for line in FileObj:
                    if line.strip() != '' and line.strip()[0] != '#':
                        datapoint = line.rstrip('\n').strip().split()
                        wave.append(float(datapoint[0].replace('D', 'E')))
                        flux.append(float(datapoint[1].replace('D', 'E')))

            wave = np.array(wave)
            flux = np.array(flux)

        return wave, flux

    def read_superfit_template(self):
        wave, flux = self.read_dat_file()
        tType = os.path.split(os.path.split(self.filename)[0])[-1]  # Name of directory is the type name
        filename = os.path.basename(self.filename)
        snName, ageInfo = os.path.basename(filename).strip('.dat').split('.')
        if ageInfo == 'max':
            age = 0
        elif ageInfo[0] == 'm':
            age = -float(ageInfo[1:])
        elif ageInfo[0] == 'p':
            age = float(ageInfo[1:])
        else:
            raise Exception("Invalid Superfit file: {0}".format(self.filename))

        nCols = 1

        return wave, flux, nCols, [age], tType

    def file_extension(self, template=False):
        if isinstance(self.filename, (list, np.ndarray)):  # Is an Nx2 array
            wave, flux = self.filename[0], self.filename[1]
            return wave, flux
        elif hasattr(self.filename, 'read'):  # Is a file handle
            self.filename.seek(0)
            return self.read_dat_file()
        else:  # Is a filename string
            filename = os.path.basename(self.filename)
            extension = filename.split('.')[-1]

            if template is True and extension == 'dat' and len(filename.split('.')) == 3 and filename.split('.')[1][0] in ['m', 'p']:  # Check if input is a superfit template
                return self.read_superfit_template()
            elif self.filename.split('-')[0] in list(catalogDict.keys()):  # Read input from catalog
                return catalogDict[self.filename[0:3]](self.filename)
            elif extension == self.filename or extension in ['flm', 'txt', 'dat']:
                return self.read_dat_file()
            elif extension == 'fits':
                return self.read_fits_file()
            elif extension == 'lnw':
                return self.snid_template_spectra_all()
            else:
                try:
                    return self.read_dat_file()
                except:
                    print("Invalid Input File")
                    return 0

    def two_col_input_spectrum(self, wave, flux, z):
        wave, flux = self.processingTools.deredshift_spectrum(wave, flux, z)

        mask = (wave >= self.w0) & (wave < self.w1)
        wave = wave[mask]
        flux = flux[mask]

        if not wave.any():
            raise Exception("The spectrum {0} with redshift {1} is out of the wavelength range {2}A to {3}A, "
                            "and cannot be classified. Please remove this object or change the input redshift of this"
                            " spectrum.".format(self.filename, z, int(self.w0), int(self.w1)))

        fluxNorm = (flux - min(flux)) / (max(flux) - min(flux))

        return wave, fluxNorm

    def snid_template_spectra_all(self):
        """lnw file"""
        with open(self.filename, 'r') as FileObj:
            for lineNum, line in enumerate(FileObj):
                # Read Header Info
                if lineNum == 0:
                    header = (line.strip('\n')).split(' ')
                    header = [x for x in header if x != '']
                    numAges, nwx, w0x, w1x, mostKnots, tname, dta, ttype, ittype, itstype = header
                    numAges, mostKnots = map(int, (numAges, mostKnots))
                    nk = np.zeros(numAges)
                    fmean = np.zeros(numAges)
                    xk = np.zeros((mostKnots, numAges))
                    yk = np.zeros((mostKnots, numAges))

                # Read Spline Info
                elif lineNum == 1:
                    splineInfo = (line.strip('\n')).split(' ')
                    splineInfo = [x for x in splineInfo if x != '']
                    for j in range(numAges):
                        nk[j], fmean[j] = (splineInfo[2 * j + 1], splineInfo[2 * j + 2])
                elif lineNum in range(2, mostKnots + 2):
                    splineInfo = (line.strip('\n')).split(' ')
                    splineInfo = [x for x in splineInfo if x != '']
                    for j in range(numAges):
                        xk[lineNum - 2, j], yk[lineNum - 2, j] = (splineInfo[2 * j + 1], splineInfo[2 * j + 2])

                elif lineNum == mostKnots + 2:
                    break

        splineInfo = (nk, fmean, xk, yk)

        # Read Normalized spectra
        if USE_PANDAS is True:
            arr = pd.read_csv(self.filename, skiprows=mostKnots + 2, header=None, delim_whitespace=True).values
        else:
            arr = np.loadtxt(self.filename, skiprows=mostKnots + 2)
        ages = arr[0]
        ages = np.delete(ages, 0)
        arr = np.delete(arr, 0, 0)

        wave = arr[:, 0]
        fluxes = np.zeros(shape=(numAges, len(arr)))  # initialise 2D array

        for i in range(0, len(arr[0]) - 1):
            fluxes[i] = arr[:, i + 1]

        if ttype == 'Ia-99aa':
            ttype = 'Ia-91T'

        return wave, fluxes, numAges, ages, ttype, splineInfo

    def snid_template_undo_processing(self, wave, flux, splineInfo, ageIdx):
        # Undo continuum removal -> then add galaxy -> then redshift
        nkAll, fmeanAll, xkAll, ykAll = splineInfo
        nk, fmean, xk, yk = int(nkAll[ageIdx]), fmeanAll[ageIdx], xkAll[:, ageIdx], ykAll[:, ageIdx]
        xk, yk = xk[:nk], yk[:nk]

        # NEED TO USE THIS TO ACTUALLY ADD THE SPLINE CONTINUUM BACK. NOT DOING ANYTHING AT THE MOMENT.

        return wave, flux


class PreProcessSpectrum(object):
    def __init__(self, w0, w1, nw):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.dwlog = np.log(w1 / w0) / nw
        self.processingTools = ProcessingTools()

    def log_wavelength(self, wave, flux):
        # Set up log wavelength array bins
        wlog = self.w0 * np.exp(np.arange(0, self.nw) * self.dwlog)

        fluxOut = self._vectorised_log_binning(wave, flux)
        # fluxOut = self._original_log_binning(wave, flux)

        minIndex, maxIndex = self.processingTools.min_max_index(fluxOut, outerVal=0)

        return wlog, fluxOut, minIndex, maxIndex

    def _vectorised_log_binning(self, wave, flux):
        """ Vectorised code version of the self._original_log_binning (for improved speed since this is the most called
        function in the script during training). This is complicated, but it has been tested to match the slower
        looping method """

        spec = np.array([wave, flux]).T
        mask = (wave >= self.w0) & (wave < self.w1)
        spec = spec[mask]
        wave, flux = spec.T
        try:
            fluxOut = np.zeros(int(self.nw))
            waveMiddle = wave[1:-1]
            waveTake1Index = wave[:-2]
            wavePlus1Index = wave[2:]
            s0List = 0.5 * (waveTake1Index + waveMiddle)
            s1List = 0.5 * (waveMiddle + wavePlus1Index)
            s0First = 0.5 * (3 * wave[0] - wave[1])
            s0Last = 0.5 * (wave[-2] + wave[-1])
            s1First = 0.5 * (wave[0] + wave[1])
            s1Last = 0.5 * (3 * wave[-1] - wave[-2])
            s0List = np.concatenate([[s0First], s0List, [s0Last]])
            s1List = np.concatenate([[s1First], s1List, [s1Last]])
            s0LogList = np.log(s0List / self.w0) / self.dwlog + 1
            s1LogList = np.log(s1List / self.w0) / self.dwlog + 1
            dnuList = s1List - s0List

            s0LogListInt = s0LogList.astype(int)
            s1LogListInt = s1LogList.astype(int)
            numOfJLoops = s1LogListInt - s0LogListInt
            jIndexes = np.flatnonzero(numOfJLoops)
            jIndexVals = s0LogListInt[jIndexes]
            prependZero = jIndexVals[0] if jIndexVals[0] < 0 else False
            if prependZero is not False:
                jIndexVals[0] = 0
                numOfJLoops[0] += prependZero
            numOfJLoops = (numOfJLoops[jIndexes])[jIndexVals < self.nw]
            fluxValList = ((flux * 1 / (s1LogList - s0LogList) * dnuList)[jIndexes])[jIndexVals < self.nw]
            fluxValList = np.repeat(fluxValList, numOfJLoops)
            minJ = min(jIndexVals)
            maxJ = (max(jIndexVals)+numOfJLoops[-1]) if (max(jIndexVals)+numOfJLoops[-1] < self.nw) else self.nw
            fluxOut[minJ:maxJ] = fluxValList[:(maxJ-minJ)]

            return fluxOut
        except Exception as e:
            print(e)
            print('wave', wave)
            print('flux', flux)
            print("########################################ERROR#######################################\n\n\n\n")
            return np.zeros(self.nw)

    def _original_log_binning(self, wave, flux):
        """ Rebin wavelengths: adapted from SNID apodize.f subroutine rebin() """
        fluxOut = np.zeros(int(self.nw))

        for i in range(0, len(wave)):
            if i == 0:
                s0 = 0.5 * (3 * wave[i] - wave[i + 1])
                s1 = 0.5 * (wave[i] + wave[i + 1])
            elif i == len(wave) - 1:
                s0 = 0.5 * (wave[i - 1] + wave[i])
                s1 = 0.5 * (3 * wave[i] - wave[i - 1])
            else:
                s0 = 0.5 * (wave[i - 1] + wave[i])
                s1 = 0.5 * (wave[i] + wave[i + 1])

            s0log = np.log(s0 / self.w0) / self.dwlog + 1
            s1log = np.log(s1 / self.w0) / self.dwlog + 1
            dnu = s1 - s0

            for j in range(int(s0log), int(s1log)):
                if j < 0 or j >= self.nw:
                    continue
                alen = 1  # min(s1log, j+1) - max(s0log, j)
                fluxval = flux[i] * alen / (s1log - s0log) * dnu
                fluxOut[j] = fluxOut[j] + fluxval

        return fluxOut

    def spline_fit(self, wave, flux, numSplinePoints, minindex, maxindex):
        continuum = np.zeros(int(self.nw)) + 1
        if (maxindex - minindex) > 5:
            spline = UnivariateSpline(wave[minindex:maxindex + 1], flux[minindex:maxindex + 1], k=3)
            splineWave = np.linspace(wave[minindex], wave[maxindex], num=numSplinePoints, endpoint=True)
            splinePoints = spline(splineWave)

            splineMore = UnivariateSpline(splineWave, splinePoints, k=3)
            splinePointsMore = splineMore(wave[minindex:maxindex + 1])

            continuum[minindex:maxindex + 1] = splinePointsMore
        else:
            print("WARNING: LESS THAN 6 POINTS IN SPECTRUM")

        return continuum

    def continuum_removal(self, wave, flux, numSplinePoints, minIndex, maxIndex):
        flux = flux + 1  # Important to keep this as +1
        contRemovedFlux = np.copy(flux)

        splineFit = self.spline_fit(wave, flux, numSplinePoints, minIndex, maxIndex)
        contRemovedFlux[minIndex:maxIndex + 1] = flux[minIndex:maxIndex + 1] / splineFit[minIndex:maxIndex + 1]
        contRemovedFluxNorm = normalise_spectrum(contRemovedFlux - 1)
        contRemovedFluxNorm = zero_non_overlap_part(contRemovedFluxNorm, minIndex, maxIndex)

        return contRemovedFluxNorm, splineFit - 1

    def mean_zero(self, flux, minindex, maxindex):
        """mean zero flux"""
        meanflux = np.mean(flux[minindex:maxindex])
        meanzeroflux = flux - meanflux
        meanzeroflux[0:minindex] = flux[0:minindex]
        meanzeroflux[maxindex + 1:] = flux[maxindex + 1:]

        return meanzeroflux

    def apodize(self, flux, minindex, maxindex, outerVal=0):
        """apodize with 5% cosine bell"""
        percent = 0.05
        fluxout = np.copy(flux) - outerVal

        nsquash = int(self.nw * percent)
        for i in range(0, nsquash):
            arg = np.pi * i / (nsquash - 1)
            factor = 0.5 * (1 - np.cos(arg))
            if (minindex + i < self.nw) and (maxindex - i >= 0):
                fluxout[minindex + i] = factor * fluxout[minindex + i]
                fluxout[maxindex - i] = factor * fluxout[maxindex - i]
            else:
                print("INVALID FLUX IN PREPROCESSING.PY APODIZE()")
                print("MININDEX=%d, i=%d" % (minindex, i))
                break

        if outerVal != 0:
            fluxout = fluxout + outerVal
            fluxout = zero_non_overlap_part(fluxout, minindex, maxindex, outerVal=outerVal)

        return fluxout


if __name__ == '__main__':
    # Plot comparison of superfit galaxies and Bsnip galaxies
    from astrodash.array_tools import normalise_spectrum
    import matplotlib.pyplot as plt

    preProcess = PreProcessSpectrum(3500, 10000, 1024)

    for gal in ['E', 'S0', 'Sa', 'Sb', 'Sc', 'SB1', 'SB2', 'SB3', 'SB4', 'SB5', 'SB6']:
        sfReadSpectrum = ReadSpectrumFile('../templates/superfit_templates/gal/%s' % gal, 3500, 10000, 1024)
        sfWave, sfFlux = sfReadSpectrum.file_extension()
        sfWave, sfFlux = sfReadSpectrum.two_col_input_spectrum(sfWave, sfFlux, z=0)
        # plt.plot(sfWave, sfFlux)
        sfWave, sfFlux, sfMinIndex, sfMaxIndex = preProcess.log_wavelength(sfWave, sfFlux)
        plt.plot(sfWave, normalise_spectrum(sfFlux))
        sfFlux, continuum = preProcess.continuum_removal(sfWave, sfFlux, 13, sfMinIndex, sfMaxIndex)
        # plt.plot(sfWave, continuum-1)

        snidReadSpectrum = ReadSpectrumFile('../templates/bsnip_v7_snid_templates/kc%s.lnw' % gal, 3500, 10000, 1024)
        snidSpectrum = snidReadSpectrum.file_extension()
        snidWave, snidFluxes, ncols, ages, ttype, splineInfo = snidSpectrum
        snidWave, snidFlux, snidMinIndex, snidMaxIndex = preProcess.log_wavelength(snidWave, snidFluxes[0])
        snidFlux = normalise_spectrum(snidFlux)
        fluxNorm = zero_non_overlap_part(snidFlux, snidMinIndex, snidMaxIndex, outerVal=0.5)

        plt.title(gal)
        plt.plot(sfWave, sfFlux, label='superfit')
        # plt.plot(snidWave, snidFlux, label='BSNIP')
        plt.plot(sfWave, normalise_spectrum(sfFlux * (continuum - 1)), label='superfit_continuumMultiply')
        plt.legend()
        plt.show()
