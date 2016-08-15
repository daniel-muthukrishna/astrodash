import numpy as np
from specutils.io import read_fits

class Redshifting(object):
    def __init__(self, wave, flux, z):
        self.wave = wave
        self.flux = flux
        self.z = z

    def redshift_spectrum(self):
        wave_new = self.wave * (self.z + 1)

        return wave_new, self.flux


class ReadInputSpectra(object):

    def __init__(self, filename, w0, w1, z):
        self.filename = filename
        self.w0 = w0
        self.w1 = w1
        self.z = z

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
            with open(self.filename) as FileObj:
                for line in FileObj:
                    datapoint = line.rstrip('\n').strip().split()
                    if (len(datapoint) >= 2):
                        wave.append(float(datapoint[0].replace('D', 'E')))
                        flux.append(float(datapoint[1].replace('D', 'E')))
        except ValueError:
            print ("Invalid Superfit file: " + self.filename) #D-13 instead of E-13

        wave = np.array(wave)
        flux = np.array(flux)

        return wave, flux


    def two_col_input_spectrum(self):
        extension = self.filename.split('.')[-1]

        if extension == 'dat':
            wave, flux = self.read_dat_file()
        elif extension == 'fits':
            wave, flux = self.read_fits_file()
        else:
            print("Invalid Input File")
            
        redshifting = Redshifting(wave, flux, self.z)
        wave, flux = redshifting.redshift_spectrum()

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
        with open(self.filename) as FileObj:
        #    text = f.readlines()
            linecount = 0
            for lines in FileObj:
                if linecount == 0:
                    header = (lines.strip('\n')).split(' ')
                    header = [x for x in header if x != '']
                    nepoch, nwx, w0x, w1x, mostknots, tname, dta, ttype, ittype, itstype = header
                if lines[1] != ' ':
                    break
                linecount += 1

        arr=np.loadtxt(self.filename, skiprows=linecount-1)
        ages = arr[0]
        ages = np.delete(ages, 0)
        arr = np.delete(arr, 0 ,0)

        wave = arr[:,0]
        flux = np.zeros(shape=(len(ages),len(arr))) # initialise 2D array

        for i in range(0, len(arr[0])-1):
            flux[i] = arr[:,i+1]


        redshifting = Redshifting(wave, flux, self.z)
        wave, flux = redshifting.redshift_spectrum()

        return wave, flux, len(ages), ages, ttype



    def snid_template_spectra(self, ageidx):
        #loop over each age instead of just first age flux
        wave, flux, ncols, ages, ttype = self.snid_template_spectra_all()
        return wave, flux[ageidx], ncols, ages, ttype

    

class PreProcessSpectrum(object):
    def __init__(self, w0, w1, nw):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.dwlog = np.log(self.w1 / self.w0) / self.nw

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
            if (i == 0):
                s0 = 0.5*(3*wave[i] - wave[i+1])
                s1 = 0.5*(wave[i] + wave[i+1])
            elif (i == len(wave) - 1):
                s0 = 0.5*(wave[i-1] + wave[i])
                s1 = 0.5*(3*wave[i] - wave[i-1])
            else:
                s0 = 0.5 * (wave[i-1] + wave[i])
                s1 = 0.5 * (wave[i] + wave[i+1])

            s0log = np.log(s0/self.w0)/self.dwlog + 1
            s1log = np.log(s1/self.w0)/self.dwlog + 1
            dnu = s1-s0

            for j in range(int(s0log), int(s1log)):
                if (j < 1 or j >= self.nw):
                    continue
                alen = 1#min(s1log, j+1) - max(s0log, j)
                fluxval = flux[i] * alen/(s1log-s0log) * dnu
                fluxout[j] = fluxout[j] + fluxval



    ##            print (j, range(int(s0log), int(s1log)), int(s0log), s0log, int(s1log), s1log)
    ##            print fluxout[j]
    ##            print (j+1, s1log, j, s0log)
    ##            print (min(s1log, j+1), max(s0log, j), alen, s1log-s0log)
    ##            print ('--------------------------')

        # Find min and max index of range
        minindex, maxindex = (0, len(wlog)-1)
        zeros = np.where(fluxout == 0)[0]
        j = 0
        for i in zeros:
            if (i != j):
                minindex = j
                maxindex = i
                break
            j += 1
        j = int(self.nw) - 1
        for i in zeros[::-1]:
            if (i != j):
                maxindex = j
                break
            j -= 1


        return (wlog, fluxout, minindex, maxindex)

    def poly_fit(self, wave, flux, order, minindex, maxindex):
        polyCoeff = np.polyfit(wave[minindex:maxindex], flux[minindex:maxindex], order)
        p = np.poly1d(polyCoeff)

        x = np.linspace(min(wave), max(wave), 100)
        y = p(wave)

        return (wave, y)

    def continuum_removal(self, wave, flux, order, minindex, maxindex):
        """Fit polynomial"""

        polyx, polyy = self.poly_fit(wave, flux, order, minindex, maxindex)

        newflux = flux - polyy

        # don't do subtraction where the data has not begun and the flux is still zero
        for i in range(0,minindex):
            newflux[i] = 0
        for i in range(maxindex,len(flux)):
            newflux[i] = 0

        return (newflux)

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
            fluxout[minindex+i] = factor*fluxout[minindex+i]
            fluxout[maxindex-i] = factor*fluxout[maxindex-i]

        return fluxout


