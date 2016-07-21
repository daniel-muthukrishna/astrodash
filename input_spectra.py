import numpy as np

def input_spectra(filename, w0, w1, z):
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

    wave, flux = redshift_spectrum(wave, flux, z)

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

class PreProcessing(object):
    """ Pre-processes spectra for cross correlation """


    def processed_data(self, filename, w0, w1, nw, z):
        polyorder = 4
        
        wave, flux = input_spectra(filename, w0, w1, z)
        binnedwave, binnedflux, minindex, maxindex = log_wavelength(wave, flux, w0, w1, nw)
        polyx, polyy = poly_fit(binnedwave, binnedflux, polyorder, minindex, maxindex)
        newflux = continuum_removal(binnedwave, binnedflux, polyorder, minindex, maxindex)
        meanzero = mean_zero(binnedwave, newflux, minindex, maxindex)
        apodized = apodize(binnedwave, meanzero, minindex, maxindex)

        return binnedwave, apodized, minindex, maxindex


N = 1024
ntypes = 14
w0 = 2500. #wavelength range in Angstroms
w1 = 11000.
nw = 1024. #number of wavelength bins

def temp_list(tempfilelist):
    f = open(tempfilelist, 'rU')

    filelist = f.readlines()
    for i in range(0,len(filelist)):
        filelist[i] = filelist[i].strip('\n')

    f.close()

    return filelist

def label_array(ttype):
    labelarray = np.zeros(ntypes)
    if (ttype == 'Ia-norm'):
        labelarray[0] = 1
        
    elif (ttype == 'IIb'):
        labelarray[1] = 1
        
    elif (ttype == 'Ia-pec'):
        labelarray[2] = 1
        
    elif (ttype == 'Ic-broad'):
        labelarray[3] = 1
        
    elif (ttype == 'Ia-csm'):
        labelarray[4] = 1
        
    elif (ttype == 'Ic-norm'):
        labelarray[5] = 1
        
    elif (ttype == 'IIP'):
        labelarray[6] = 1
        
    elif (ttype == 'Ib-pec'):
        labelarray[7] = 1
        
    elif (ttype == 'IIL'):
        labelarray[8] = 1
        
    elif (ttype == 'Ib-norm'):
        labelarray[9] = 1
        
    elif (ttype == 'Ia-91bg'):
        labelarray[10] = 1
        
    elif (ttype == 'II-pec'):
        labelarray[11] = 1
        
    elif (ttype == 'Ia-91T'):
        labelarray[12] = 1
        
    elif (ttype == 'IIn'):
        labelarray[13] = 1
        
    elif (ttype == 'Ia'):
        labelarray[0] = 1
        
    elif (ttype == 'Ib'):
        labelarray[9] = 1
        
    elif (ttype == 'Ic'):
        labelarray[5] = 1
        
    elif (ttype == 'II'):
        labelarray[13] = 1
    else:
        print ("Invalid type")

        
    return labelarray


def sf_age(filename):
    snName, extension = filename.strip('.dat').split('.')
    ttype, snName = snName.split('/')
    
    if (extension == 'max'):
        age = 0
    elif (extension[0] == 'p'):
        age = float(extension[1:])
    elif (extension[0] == 'm'):
        age = -float(extension[1:])
    else:
        print "Invalid Superfit Filename: " + filename

    return snName, ttype, age

def superfit_template_data(filename, z):
    """ Returns wavelength and flux after all preprocessing """
    data = PreProcessing()
    wave, flux, minIndex, maxIndex = data.processed_data(sfTemplateLocation + filename, w0, w1, nw, z)
    snName, ttype, age = sf_age(filename)

    print snName, ttype, age
    
    return wave, flux, minIndex, maxIndex, age, snName, ttype

def superfit_templates_to_arrays(sftempfilelist):
    templist = temp_list(sftempfilelist)
    images = np.empty((0,N), np.float32) #Number of pixels
    labels = np.empty((0,ntypes), float) #Number of labels (SN types)
    filenames = []
    
    for i in range(0, len(templist)):
        tempwave, tempflux, tminindex, tmaxindex, age, snName, ttype = superfit_template_data(templist[i])
        label = label_array(ttype)
        
        if ((float(age) < 4 and float(age) > -4)):                
            nonzeroflux = tempflux[tminindex:tmaxindex+1]
            newflux = (nonzeroflux - min(nonzeroflux))/(max(nonzeroflux)-min(nonzeroflux))
            newflux2 = np.concatenate((tempflux[0:tminindex], newflux, tempflux[tmaxindex+1:]))
            images = np.append(images, np.array([newflux2]), axis=0) #images.append(newflux2)
            labels = np.append(labels, np.array([label]), axis=0) #labels.append(ttype)
            filenames.append(templist[i])
            
    return images, labels, np.array(filenames)

###########################################################

def redshift_spectrum(wave, flux, z):
    wave_new = wave*(z+1)

    return wave_new, flux



sfTemplateLocation = '/home/dan/Desktop/SNClassifying_DeepLearning/templates/superfit_templates/sne/'
filename = 'Ia/sn1981b.max.dat'
images = np.empty((0,N), np.float32) #Number of pixels
labels = np.empty((0,ntypes), float) #Number of labels (SN types)
filenames = []
redshifts = []
    
for z in np.linspace(0, 0.5, 501):
    tempwave, tempflux, tminindex, tmaxindex, age, snName, ttype = superfit_template_data(filename, z)
    label = label_array(ttype)
    nonzeroflux = tempflux[tminindex:tmaxindex+1]
    newflux = (nonzeroflux - min(nonzeroflux))/(max(nonzeroflux)-min(nonzeroflux))
    newflux2 = np.concatenate((tempflux[0:tminindex], newflux, tempflux[tmaxindex+1:]))
    images = np.append(images, np.array([newflux2]), axis=0) #images.append(newflux2)
    labels = np.append(labels, np.array([label]), axis=0) #labels.append(ttype)
    filenames.append(filename + "_"+ str(z))
    redshifts.append(z)

inputImages = np.array(images)
inputLabels = np.array(labels)
inputFilenames = np.array(filenames)
inputRedshifts = np.array(redshifts)

np.savez_compressed('input_data.npz', inputImages = inputImages, inputLabels=inputLabels, inputFilenames=inputFilenames, inputRedshifts=inputRedshifts)
