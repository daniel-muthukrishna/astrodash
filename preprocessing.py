import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

c = 3e8
filelocation = 'C:\Users\Daniel\OneDrive\Documents\Thesis Project\superfit\sne\Ia\\'
snfilename ='sn2002bo.m01.dat' #sn2003jo.dat
templatefilename = 'sn1999ee.m08.dat'

# w0test = 2500. #wavelength range in Angstroms
# w1test = 11000.
# nwtest = 1024. #number of wavelength bins

def input_spectra(filename, w0, w1):
    """
    Takes a two column file of wavelength and flux as an input
    and returns the numpy arrays of the wavelength and normalised flux.
    """
    wave = []
    flux = []
    try:
        with open(filename) as FileObj:
            for line in FileObj:
                datapoint = line.rstrip('\n').strip().split()
                if (len(datapoint) >= 2):
                    wave.append(float(datapoint[0].replace('D', 'E')))
                    flux.append(float(datapoint[1].replace('D', 'E')))
    except ValueError:
        print ("Invalid Superfit file: " + filename) #D-13 instead of E-13

    wave = np.array(wave)
    flux = np.array(flux)

    if max(wave) >= w1:
        for i in range(0,len(wave)):
            if (wave[i] >= w1):
                break
        wave = wave[0:i]
        flux = flux[0:i]
    if min(wave) < w0:
        for i in range(0,len(wave)):
            if (wave[i] >= w0):
                break
        wave = wave[i:]
        flux = flux[i:]
                
    fluxNorm = (flux - min(flux))/(max(flux)-min(flux))
    
    return wave, fluxNorm

def template_spectra_all(filename):
    """lnw file"""
    with open(filename) as FileObj:
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

    arr=np.loadtxt(filename, skiprows=linecount-1)
    ages = arr[0]
    ages = np.delete(ages, 0)
    arr = np.delete(arr, 0 ,0)

    wave = arr[:,0]
    flux = np.zeros(shape=(len(ages),len(arr))) # initialise 2D array
    
    for i in range(0, len(arr[0])-1):
        flux[i] = arr[:,i+1]
    
    return wave, flux, len(ages), ages, ttype

##
##def store_files(templist):
##    for filename in templist:
##        

def template_spectra(filename, ageidx):
    #loop over each age instead of just first age flux
    wave, flux, ncols, ages, ttype = template_spectra_all(filename)
    return wave, flux[ageidx], ncols, ages, ttype

    


def log_wavelength(wave, flux, w0, w1, nw):

    fluxout = np.zeros(int(nw))
    j =0
    # Set up log wavelength array bins
    dwlog = np.log(w1/w0)/nw
    wlog = w0 * np.exp(np.arange(0,nw) * dwlog)

    A = nw/np.log(w1/w0)
    B = -nw*np.log(w0)/np.log(w1/w0)

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

        s0log = np.log(s0/w0)/dwlog + 1
        s1log = np.log(s1/w0)/dwlog + 1
        dnu = s1-s0
        
        for j in range(int(s0log), int(s1log)):
            if (j < 1 or j >= nw):
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
    j = int(nw) - 1
    for i in zeros[::-1]:
        if (i != j):
            maxindex = j
            break
        j -= 1
        

    return (wlog, fluxout, minindex, maxindex)

def poly_fit(wave, flux, order, minindex, maxindex):
    polyCoeff = np.polyfit(wave[minindex:maxindex], flux[minindex:maxindex], order)
    p = np.poly1d(polyCoeff)
    
    x = np.linspace(min(wave), max(wave), 100)
    y = p(wave)

    return (wave, y)

def continuum_removal(wave, flux, order, minindex, maxindex):
    """Fit polynomial"""
    
    polyx, polyy = poly_fit(wave, flux, order, minindex, maxindex)
    
    newflux = flux - polyy
    
    # don't do subtraction where the data has not begun and the flux is still zero
    for i in range(0,minindex):
        newflux[i] = 0
    for i in range(maxindex,len(flux)):
        newflux[i] = 0
        
    return (newflux)

def mean_zero(wave, flux, minindex, maxindex):
    """mean zero flux"""
    meanflux = np.mean(flux[minindex:maxindex])
    meanzeroflux = flux - meanflux

    for i in range(0,minindex):
        meanzeroflux[i] = 0
    for i in range(maxindex,len(flux)):
        meanzeroflux[i] = 0
    
    return meanzeroflux

def apodize(wave, flux, minindex, maxindex):
    """apodize with 5% cosine bell"""
    n = 1024
    percent = 0.05
    fluxout = flux + 0
    
    nsquash = int(n*percent)
    for i in range(0, nsquash):
        arg = np.pi * i/(nsquash-1)
        factor = 0.5*(1-np.cos(arg))
        fluxout[minindex+i] = factor*fluxout[minindex+i]
        fluxout[maxindex-i] = factor*fluxout[maxindex-i]
        
    return fluxout

##def preprocessing(filename):
##    """ All preprocessing """
##    polyorder = 4
##    
##    wave, flux = input_spectra(filename)
##    binnedwave, binnedflux, minindex, maxindex = log_wavelength(snwave, snflux, w0test, w1test, nwtest)
##    polyx, polyy = poly_fit(binnedwave, binnedflux, polyorder, minindex, maxindex)
##    newflux = continuum_removal(binnedwave, binnedflux, polyorder, minindex, maxindex)
##    meanzero = mean_zero(binnedwave, newflux, minindex, maxindex)
##    apodized = apodize(binnedwave, meanzero, minindex, maxindex)
##
##    return (binnedwave, apodized)
##    
##
##def xcorr(inputfilename, tempfilename):
##    """ Cross correlation """
##    inputwave, inputflux = preprocessing(inputfilename)
##    tempwave, tempflux = preprocessing(tempfilename)
##    return

"""
polyorder = 3
plt.figure('sn flux vs wavelength')
snwave, snflux = input_spectra(filelocation+snfilename)
plt.plot(snwave, snflux)
plt.xlabel(r'Wavelength $(\AA)$')
plt.ylabel('Relative Flux')

plt.figure('log binned')
binnedwave, binnedflux, minindex, maxindex = log_wavelength(snwave, snflux, w0test, w1test, nwtest)
polyx, polyy = poly_fit(binnedwave, binnedflux, polyorder, minindex, maxindex)
newflux = continuum_removal(binnedwave, binnedflux, polyorder, minindex, maxindex)
meanzero = mean_zero(binnedwave, newflux, minindex, maxindex)
apodized = apodize(binnedwave, meanzero, minindex, maxindex)
plt.plot(binnedwave, binnedflux, label='binned')
plt.plot(polyx, polyy, label='polyfit')
plt.plot(binnedwave, newflux, label='continuum subtracted')
plt.plot(binnedwave, meanzero, label='zero mean')
plt.plot(binnedwave, apodized, label = 'cosine taper')
plt.xlabel("Binned Wavelengths")
plt.ylabel('Relative Flux')
plt.legend()

plt.figure('template flux vs wavelength')
tempwave, tempflux = input_spectra(filelocation+templatefilename)
plt.plot(tempwave, tempflux)
plt.xlabel(r'Wavelength $(\AA)$')
plt.ylabel('Relative Flux')


plt.figure('Fourier Transform')
N=600
T=1./800.
yf = fft(apodized)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]), 'b.')
plt.xlabel(r'Wave number, k')
plt.ylabel('Fourier Transform of flux')

plt.figure()
wd,fd = preprocessing(filelocation+snfilename)
wt,ft = preprocessing(filelocation+templatefilename)
plt.plot(wd,fd)
plt.plot(wt,ft)
plt.show()
"""

# Compute cross-correlation
##plt.figure('xcorr')
##plt.plot(np.correlate(snflux, tempflux))
##

##from PyAstronomy import pyasl
##plt.figure('xcorr')
##rv, cc = pyasl.crosscorrRV(snwave, snflux, tempwave, tempflux, -30, 30, 0.02, skipedge=1)
##maxind = np.argmax(cc)
##print "Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s"
##if rv[maxind] > 0.0:
##  print "  A red-shift with respect to the template"
##else:
##  print "  A blue-shift with respect to the template"
##
##plt.plot(rv, cc, 'bp-')
##plt.plot(rv[maxind], cc[maxind], 'ro')



#plt.show()

