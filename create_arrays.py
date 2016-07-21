import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sn_processing import PreProcessing

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


def snid_template_data(filename, ageidx):
    """ lnw template files """
    data = PreProcessing()
    wave, flux, ncols, ages, ttype, minindex, maxindex = data.templates(snidTemplateLocation+filename, ageidx, w0, w1, nw)

    return wave, flux, ncols, ages, ttype, minindex, maxindex

def age_bin(age, ageBinSize, minAge, maxAge):
    #print(i, int(i/5), int(i/5)+ int(33/5), int((i + 33)/5))
    ageBin = int(age/ageBinSize) - int(minAge/ageBinSize) #around zero has double bin siz

    return ageBin

def age_labels(ageBinSize, minAge, maxAge):
    ageLabels = []

    ageBinPrev = 0
    ageLabelMin = minAge
    for age in np.arange(minAge, maxAge):
        ageBin = age_bin(age, ageBinSize, minAge, maxAge)
        
        if (ageBin != ageBinPrev):
            ageLabelMax = age
            ageLabels.append(str(ageLabelMin) + " to " + str(ageLabelMax))
            ageLabelMin = ageLabelMax
            
        ageBinPrev = ageBin
            
        
    ageLabels.append(str(ageLabelMin) + " to " + str(maxAge))   
    
    return ageLabels
    

minAge = -50
maxAge = 50
ageBinSize = 4.
numOfAgeBins = age_bin(maxAge,ageBinSize, minAge, maxAge) + 1
nLabels = ntypes*numOfAgeBins
ageLabels = age_labels(ageBinSize, minAge, maxAge)


def label_array(ttype, age):
    ageBin = age_bin(age, ageBinSize, minAge, maxAge)
    labelarray = np.zeros((ntypes, numOfAgeBins))
    typeNames = []
    
    if (ttype == 'Ia-norm'):
        typeIndex = 0

    elif (ttype == 'IIb'):
        typeIndex = 1
        
    elif (ttype == 'Ia-pec'):
        typeIndex = 2
        
    elif (ttype == 'Ic-broad'):
        typeIndex = 3
        
    elif (ttype == 'Ia-csm'):
        typeIndex = 4
        
    elif (ttype == 'Ic-norm'):
        typeIndex = 5
        
    elif (ttype == 'IIP'):
        typeIndex = 6
        
    elif (ttype == 'Ib-pec'):
        typeIndex = 7
        
    elif (ttype == 'IIL'):
        typeIndex = 8
        
    elif (ttype == 'Ib-norm'):
        typeIndex = 9
        
    elif (ttype == 'Ia-91bg'):
        typeIndex = 10
        
    elif (ttype == 'II-pec'):
        typeIndex = 11
        
    elif (ttype == 'Ia-91T'):
        typeIndex = 12
        
    elif (ttype == 'IIn'):
        typeIndex = 13
        
    elif (ttype == 'Ia'):
        typeIndex = 0
        
    elif (ttype == 'Ib'):
        typeIndex = 9
        
    elif (ttype == 'Ic'):
        typeIndex = 5
        
    elif (ttype == 'II'):
        typeIndex = 13
    else:
        print ("Invalid type")
        

    labelarray[typeIndex][ageBin] = 1
    labelarray = labelarray.flatten()
    
    typeNames.append(ttype + ": " + ageLabels[ageBin])
    typeNames = np.array(typeNames)
    
    
    
    return labelarray, typeNames

def snid_templates_to_arrays(tempfilelist):
    ''' This function is for the SNID processed files, which
        have been preprocessed to negatives, and so cannot be
        imaged yet '''
    #inputwave, inputflux = preproc_data(inputfilename)
    templist = temp_list(tempfilelist)
    typeList = []
    images = np.empty((0,N), np.float32) #Number of pixels
    labels = np.empty((0,nLabels), float) #Number of labels (SN types)
    filenames = []
    typeNames = []
    agesList = []

    for i in range(0, len(templist)):
        ncols = 15
        for ageidx in range(0,100):
            if (ageidx < ncols):
                tempwave, tempflux, ncols, ages, ttype, tminindex, tmaxindex = snid_template_data(templist[i], ageidx)
                agesList.append(ages[ageidx])

                if ((float(ages[ageidx]) > minAge and float(ages[ageidx]) < maxAge)):    
                    label, typeName = label_array(ttype, ages[ageidx])
                    nonzeroflux = tempflux[tminindex:tmaxindex+1]
                    newflux = (nonzeroflux - min(nonzeroflux))/(max(nonzeroflux)-min(nonzeroflux))
                    newflux2 = np.concatenate((tempflux[0:tminindex], newflux, tempflux[tmaxindex+1:]))
                    images = np.append(images, np.array([newflux2]), axis=0) #images.append(newflux2)
                    labels = np.append(labels, np.array([label]), axis=0) #labels.append(ttype)
                    filenames.append(templist[i] + '_' + ttype + '_' + str(ages[ageidx]))
                    typeNames.append(typeName)
                    
        print templist[i]
        #Create List of all SN types
        if ttype not in typeList:
            typeList.append(ttype)

                
    return typeList, images, labels, np.array(filenames), typeNames

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


def superfit_template_data(filename):
    """ Returns wavelength and flux after all preprocessing """
    data = PreProcessing()
    wave, flux, minIndex, maxIndex = data.processed_data(sfTemplateLocation + filename, w0, w1, nw)
    snName, ttype, age = sf_age(filename)

    print snName, ttype, age
    
    return wave, flux, minIndex, maxIndex, age, snName, ttype
	

def superfit_templates_to_arrays(sftempfilelist):
    templist = temp_list(sftempfilelist)
    images = np.empty((0,N), np.float32) #Number of pixels
    labels = np.empty((0,nLabels), float) #Number of labels (SN types)
    filenames = []
    typeNames = []
    
    for i in range(0, len(templist)):
        tempwave, tempflux, tminindex, tmaxindex, age, snName, ttype = superfit_template_data(templist[i])
        
        if ((float(ages[ageidx]) > minAge and float(ages[ageidx]) > maxAge)):
            label, typeName = label_array(ttype, ages[ageidx])                    
            nonzeroflux = tempflux[tminindex:tmaxindex+1]
            newflux = (nonzeroflux - min(nonzeroflux))/(max(nonzeroflux)-min(nonzeroflux))
            newflux2 = np.concatenate((tempflux[0:tminindex], newflux, tempflux[tmaxindex+1:]))
            images = np.append(images, np.array([newflux2]), axis=0) #images.append(newflux2)
            labels = np.append(labels, np.array([label]), axis=0) #labels.append(ttype)
            filenames.append(templist[i])
            typeNames.append(typeName)
            
    return images, labels, np.array(filenames), typeNames


def shuffle_arrays(images, labels, filenames, typeNames):
    imagesShuf = []
    labelsShuf = []
    filenamesShuf = []
    typeNamesShuf = []

    #Randomise order
    indexShuf = range(len(images))
    shuffle(indexShuf)
    for i in indexShuf:
        imagesShuf.append(images[i])
        labelsShuf.append(labels[i])
        filenamesShuf.append(filenames[i])
        typeNamesShuf.append(typeNames[i])

    return np.array(imagesShuf), np.array(labelsShuf), np.array(filenamesShuf), np.array(typeNamesShuf)

def count_labels(labels):
    counts = np.zeros(nLabels)

    for i in range(len(labels)):
        counts = labels[i] + counts

    return counts

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

countsBefore = np.zeros(nLabels)
countsAfter = np.zeros(nLabels)
    
"""Randomise order and shortlist arrays to have even
    number of SN for each type """
def over_sample_arrays(images, labels, filenames, typeNames):
    counts = count_labels(labels)
    print "Before OverSample" #
    print counts #
    
    overSampleAmount = div0(20*max(counts), counts) #ignore zeros in counts
    imagesOverSampled = []
    labelsOverSampled = []
    filenamesOverSampled = []
    typeNamesOverSampled = []

    counts1 = np.zeros(nLabels)

    imagesShuf, labelsShuf, filenamesShuf, typeNamesShuf = shuffle_arrays(images, labels, filenames, typeNames)

    for i in range(len(labelsShuf)):
        label = labelsShuf[i]
        image = imagesShuf[i]
        filename = filenamesShuf[i]
        typeName = typeNamesShuf[i]

        labelIndex = np.argmax(label)
        
        for r in range(int(overSampleAmount[labelIndex])):
            imagesOverSampled.append(image)
            labelsOverSampled.append(label)
            filenamesOverSampled.append(filename)
            typeNamesOverSampled.append(typeName)
            counts1 = label + counts1
    print "After OverSample" #
    print counts1 #
    countsBefore = counts #
    countsAfter = counts1 #
    #[ 372.    8.   22.   12.    1.   22.   26.    6.    1.    7.   34.    5.  44.    6.]
    imagesOverSampled = np.array(imagesOverSampled)
    labelsOverSampled = np.array(labelsOverSampled)
    filenamesOverSampled = np.array(filenamesOverSampled)
    typeNamesOverSampled = np.array(typeNamesOverSampled)
    imagesOverSampledShuf, labelsOverSampledShuf, filenamesOverSampledShuf, typeNamesOverSampledShuf = shuffle_arrays(imagesOverSampled, labelsOverSampled, filenamesOverSampled, typeNamesOverSampled)
        
    return imagesOverSampledShuf, labelsOverSampledShuf, filenamesOverSampledShuf, typeNamesOverSampledShuf
        

def all_templates_to_arrays(snidtempfilelist, sftempfilelist):
    images = np.empty((0,N), np.float32) #Number of pixels
    labels = np.empty((0,ntypes), float) #Number of labels (SN types)
    typeList = []

    typelistSnid, imagesSnid, labelsSnid, filenamesSnid, typeNamesSnid = snid_templates_to_arrays(snidtempfilelist)
    #imagesSuperfit, labelsSuperfit, filenamesSuperfit, typeNamesSuperfit = superfit_templates_to_arrays(sftempfilelist)
    
    images = np.vstack((imagesSnid))#, imagesSuperfit)) #Add in other templates from superfit etc.
    labels = np.vstack((labelsSnid))#, labelsSuperfit))
    filenames = np.hstack((filenamesSnid))#, filenamesSuperfit))
    typeNames = np.hstack((typeNamesSnid))
    
    typeList = typelistSnid
    
    
    imagesShuf, labelsShuf, filenamesShuf, typeNamesShuf = shuffle_arrays(images, labels, filenames, typeNames)#imagesShortlist, labelsShortlist = shortlist_arrays(images, labels)

    
    return typeList, imagesShuf, labelsShuf, filenamesShuf, typeNamesShuf #imagesShortlist, labelsShortlist


def sort_data(snidtempfilelist, sftempfilelist):
    trainPercentage = 0.8
    testPercentage = 0.2
    validatePercentage = 0.
    
    typeList, images, labels, filenames, typeNames = all_templates_to_arrays(snidtempfilelist, sftempfilelist)
    #imagesSuperfit, labelsSuperfit, filenamesSuperfit = superfit_templates_to_arrays(sftempfilelist)
    #imagesSuperfit, labelsSuperfit, filenamesSuperfit = shuffle_arrays(imagesSuperfit, labelsSuperfit, filenamesSuperfit)


    trainSize = int(trainPercentage*len(images))
    testSize = int(testPercentage*len(images))
    
    trainImages = images[:trainSize]
    testImages = images[trainSize : trainSize+testSize]
    validateImages = images[trainSize+testSize:]
    trainLabels = labels[:trainSize]
    testLabels = labels[trainSize : trainSize+testSize]
    validateLabels = labels[trainSize+testSize:]
    trainFilenames = filenames[:trainSize]
    testFilenames = filenames[trainSize : trainSize+testSize]
    validateFilenames = filenames[trainSize+testSize:]
    trainTypeNames = typeNames[:trainSize]
    testTypeNames = typeNames[trainSize : trainSize+testSize]
    validateTypeNames = typeNames[trainSize+testSize:]

    trainImagesOverSample, trainLabelsOverSample, trainFilenamesOverSample, trainTypeNamesOverSample = over_sample_arrays(trainImages, trainLabels, trainFilenames, trainTypeNames)
    testImagesShortlist, testLabelsShortlist, testFilenamesShortlist, testTypeNamesShortlist = testImages, testLabels, testFilenames, testTypeNames#(testImages, testLabels, testFilenames)

    return ((trainImagesOverSample, trainLabelsOverSample, trainFilenamesOverSample, trainTypeNamesOverSample), (testImagesShortlist, testLabelsShortlist, testFilenamesShortlist, testTypeNamesShortlist), (validateImages, validateLabels, validateFilenames, validateTypeNames))




snidTemplateLocation = '/home/dan/Desktop/SNClassifying/templates/'
sfTemplateLocation = '/home/dan/Desktop/SNClassifying/templates/superfit_templates/sne/'
snidtempfilelist1 = snidTemplateLocation + 'templist'
sftempfilelist1 = sfTemplateLocation + 'templist.txt'



sortData = sort_data(snidtempfilelist1, sftempfilelist1)

trainImages = sortData[0][0]
trainLabels = sortData[0][1]
trainFilenames = sortData[0][2]
trainTypeNames = sortData[0][3]
testImages = sortData[1][0]
testLabels = sortData[1][1]
testFilenames = sortData[1][2]
testTypeNames = sortData[1][3]
validateImages = sortData[2][0]
validateLabels = sortData[2][1]
validateFilenames = sortData[2][2]
validateTypeNames = sortData[2][3]

np.savez_compressed('file2.npz', trainImages=trainImages, trainLabels=trainLabels, trainFilenames=trainFilenames, trainTypeNames=trainTypeNames, testImages = testImages, testLabels=testLabels, testFilenames=testFilenames, testTypeNames=testTypeNames)




