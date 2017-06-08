import numpy as np
from specutils.io import read_fits
from scipy.interpolate import interp1d, UnivariateSpline


class ProcessingTools(object):

    def redshift_spectrum(self, wave, flux, z):
        wave_new = wave * (z + 1)

        return wave_new, flux

    def deredshift_spectrum(self, wave, flux, z):
        wave_new = wave / (z + 1)

        return wave_new, flux

    def min_max_index(self, flux):
        nw = len(flux)
        minindex, maxindex = (0, nw - 1)
        zeros = np.where(flux == 0)[0]
        j = 0
        for i in zeros:
            if (i != j):
                break
            j += 1
            minindex = j
        j = int(nw) - 1
        for i in zeros[::-1]:
            if (i != j):
                break
            j -= 1
            maxindex = j

        return minindex, maxindex


class ReadSpectrumFile(object):

    def __init__(self, filename, w0, w1, nw):
        self.filename = filename
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.processingTools = ProcessingTools()

    def read_fits_file(self):
        #filename = unicode(self.filename.toUtf8(), encoding="UTF-8")
        spectrum = read_fits.read_fits_spectrum1d(self.filename)
        wave = np.array(spectrum.wavelength)
        flux = np.array(spectrum.flux)
        flux[np.isnan(flux)] = 0 #convert nan's to zeros

        return wave, flux

    def read_dat_file(self):
        wave = []
        flux = []
        try:
            with open(self.filename, 'r') as FileObj:
                for line in FileObj:
                    if line.strip()[0] != '#':
                        datapoint = line.rstrip('\n').strip().split()
                        wave.append(float(datapoint[0].replace('D', 'E')))
                        flux.append(float(datapoint[1].replace('D', 'E')))
        except (ValueError, IndexError):
            print("Invalid Superfit file: " + self.filename) #D-13 instead of E-13

        wave = np.array(wave)
        flux = np.array(flux)

        return wave, flux

    def file_extension(self):
        extension = self.filename.split('.')[-1]
        if extension == 'dat' or extension == self.filename or extension in ['flm', 'txt', 'dat']:
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

        if max(wave) >= self.w1:
            for i in range(0,len(wave)):
                if (wave[i] >= self.w1):
                    break
            wave = wave[0:i]
            flux = flux[0:i]
        if min(wave) < self.w0:
            for i in range(0,len(wave)):
                if (wave[i] >= self.w0):
                    break
            wave = wave[i:]
            flux = flux[i:]

        fluxNorm = (flux - min(flux))/(max(flux)-min(flux))

        return wave, fluxNorm

    def snid_template_spectra_all(self):
        """lnw file"""
        with open(self.filename, 'r') as FileObj:
            for lineNum, line in enumerate(FileObj):
                # Read Header Info
                if lineNum == 0:
                    header = (line.strip('\n')).split(' ')
                    header = [x for x in header if x != '']
                    numAges, nwx, w0x, w1x, mostknots, tname, dta, ttype, ittype, itstype = header
                    numAges, mostknots = map(int, (numAges, mostknots))
                    nk = np.zeros(numAges)
                    fmean = np.zeros(numAges)
                    xk = np.zeros((mostknots,numAges))
                    yk = np.zeros((mostknots,numAges))

                # Read Spline Info
                elif lineNum == 1:
                    splineInfo = (line.strip('\n')).split(' ')
                    splineInfo = [x for x in splineInfo if x != '']
                    for j in range(numAges):
                        nk[j], fmean[j] = (splineInfo[2*j+1], splineInfo[2*j+2])
                elif lineNum in range(2, mostknots+2):
                    splineInfo = (line.strip('\n')).split(' ')
                    splineInfo = [x for x in splineInfo if x != '']
                    for j in range(numAges):
                        xk[lineNum-2,j], yk[lineNum-2,j] = (splineInfo[2*j+1], splineInfo[2*j+2])

                elif lineNum == mostknots+2:
                    break

        splineInfo = (nk, fmean, xk, yk)

        # Read Normalized spectra
        arr=np.loadtxt(self.filename, skiprows=mostknots+2)
        ages = arr[0]
        ages = np.delete(ages, 0)
        arr = np.delete(arr, 0 ,0)

        wave = arr[:,0]
        fluxes = np.zeros(shape=(numAges,len(arr))) # initialise 2D array

        for i in range(0, len(arr[0])-1):
            fluxes[i] = arr[:, i+1]

        if ttype == 'Ia-99aa':
            ttype = 'Ia-91T'

        return wave, fluxes, numAges, ages, ttype, splineInfo

    def snid_template_undo_processing(self, wave, flux, splineInfo, ageIdx):
        # Undo continuum removal -> then add galaxy -> then redshift
        nkAll, fmeanAll, xkAll, ykAll = splineInfo
        nk, fmean, xk, yk = int(nkAll[ageIdx]), fmeanAll[ageIdx], xkAll[:,ageIdx], ykAll[:,ageIdx]
        xk, yk = xk[:nk], yk[:nk]

        # NEED TO USE THIS TO ACTUALLY ADD THE SPLINE CONTINUUM BACK

        return wave, flux


class PreProcessSpectrum(object):
    def __init__(self, w0, w1, nw):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.dwlog = np.log(w1/w0) / nw
        self.processingTools = ProcessingTools()

    def log_wavelength(self, wave, flux):
        fluxout = np.zeros(int(self.nw))
        j =0
        # Set up log wavelength array bins
        wlog = self.w0 * np.exp(np.arange(0,self.nw) * self.dwlog)

        A = self.nw/np.log(self.w1/self.w0)
        B = -self.nw*np.log(self.w0)/np.log(self.w1/self.w0)

        binnedwave = A*np.log(wave) + B

        # Rebin wavelengths
        for i in range(0,len(wave)):
            if i == 0:
                s0 = 0.5*(3*wave[i] - wave[i+1])
                s1 = 0.5*(wave[i] + wave[i+1])
            elif i == len(wave) - 1:
                s0 = 0.5*(wave[i-1] + wave[i])
                s1 = 0.5*(3*wave[i] - wave[i-1])
            else:
                s0 = 0.5 * (wave[i-1] + wave[i])
                s1 = 0.5 * (wave[i] + wave[i+1])

            s0log = np.log(s0/self.w0)/self.dwlog + 1
            s1log = np.log(s1/self.w0)/self.dwlog + 1
            dnu = s1-s0

            for j in range(int(s0log), int(s1log)):
                if j < 1 or j >= self.nw:
                    continue
                alen = 1#min(s1log, j+1) - max(s0log, j)
                fluxval = flux[i] * alen/(s1log-s0log) * dnu
                fluxout[j] = fluxout[j] + fluxval



    ##            print(j, range(int(s0log), int(s1log)), int(s0log), s0log, int(s1log), s1log)
    ##            print(fluxout[j])
    ##            print(j+1, s1log, j, s0log)
    ##            print(min(s1log, j+1), max(s0log, j), alen, s1log-s0log)
    ##            print('--------------------------')


        minindex, maxindex = self.processingTools.min_max_index(fluxout)
        minindex, maxindex = int(minindex), int(maxindex)


        return wlog, fluxout, minindex, maxindex

    def spline_fit(self, wave, flux, numSplinePoints, minindex, maxindex):
        continuum = np.zeros(int(self.nw))
        minindex, maxindex = int(minindex), int(maxindex)
        if (maxindex - minindex) > 5:
            spline = UnivariateSpline(wave[minindex:maxindex+1], flux[minindex:maxindex+1], k=3)
            splineWave = np.linspace(wave[minindex], wave[maxindex], num=numSplinePoints, endpoint=True)
            splinePoints = spline(splineWave)

            splineMore = UnivariateSpline(splineWave, splinePoints, k=3)
            splinePointsMore = splineMore(wave[minindex:maxindex])

            continuum[minindex:maxindex] = splinePointsMore
        else:
            print("WARNING: LESS THAN 6 POINTS IN SPECTRUM")

        return continuum

    def continuum_removal(self, wave, flux, numSplinePoints, minindex, maxindex):
        newflux = np.zeros(int(self.nw))

        splineFit = self.spline_fit(wave, flux, numSplinePoints, minindex, maxindex)
        newflux[minindex:maxindex] = flux[minindex:maxindex] - splineFit[minindex:maxindex]

        return newflux, splineFit

    def mean_zero(self, wave, flux, minindex, maxindex):
        """mean zero flux"""
        meanflux = np.mean(flux[minindex:maxindex])
        meanzeroflux = flux - meanflux

        for i in range(0,minindex):
            meanzeroflux[i] = 0
        for i in range(maxindex,len(flux)):
            meanzeroflux[i] = 0

        return meanzeroflux

    def apodize(self, wave, flux, minindex, maxindex):
        """apodize with 5% cosine bell"""
        percent = 0.05
        fluxout = flux + 0

        nsquash = int(self.nw*percent)
        for i in range(0, nsquash):
            arg = np.pi * i/(nsquash-1)
            factor = 0.5*(1-np.cos(arg))
            if (minindex + i < self.nw) and (maxindex - i >= 0):
                fluxout[minindex+i] = factor*fluxout[minindex+i]
                fluxout[maxindex-i] = factor*fluxout[maxindex-i]
            else:
                print("INVALID FLUX IN PREPROCESSING.PY APODIZE()")
                print("MININDEX=%d, i=%d" % (minindex, i))
                break

        return fluxout
