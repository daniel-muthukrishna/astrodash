import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sn_processing import PreProcessing

N = 1024
ntypes = 14

w0 = 2500. #wavelength range in Angstroms
w1 = 11000.
nw = 1024. #number of wavelength bins

snidTemplateLocation = '/home/dan/Desktop/SNClassifying/templates/'
sfTemplateLocation = '/home/dan/Desktop/SNClassifying/templates/superfit_templates/sne/'
snidtempfilelist1 = snidTemplateLocation + 'templist'
sftempfilelist1 = sfTemplateLocation + 'templist.txt'

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

def snid_templates_to_arrays(tempfilelist):
    ''' This function is for the SNID processed files, which
        have been preprocessed to negatives, and so cannot be
        imaged yet '''
    #inputwave, inputflux = preproc_data(inputfilename)
    templist = temp_list(tempfilelist)
    typeList = []
    images = np.empty((0,N), np.float32) #Number of pixels
    labels = np.empty((0,ntypes), float) #Number of labels (SN types)
    filenames = []

    for i in range(0, len(templist)):
        ncols = 15
        for ageidx in range(0,100):
            if (ageidx < ncols):
                tempwave, tempflux, ncols, ages, ttype, tminindex, tmaxindex = snid_template_data(templist[i], ageidx)
                label = label_array(ttype)
                
                #Check if age is 0ish and create list of lists for images and labels
                if ((float(ages[ageidx]) < 4 and float(ages[ageidx]) > -4)):		
                    
                    nonzeroflux = tempflux[tminindex:tmaxindex+1]
                    newflux = (nonzeroflux - min(nonzeroflux))/(max(nonzeroflux)-min(nonzeroflux))
                    newflux2 = np.concatenate((tempflux[0:tminindex], newflux, tempflux[tmaxindex+1:]))
                    images = np.append(images, np.array([newflux2]), axis=0) #images.append(newflux2)
                    labels = np.append(labels, np.array([label]), axis=0) #labels.append(ttype)
                    filenames.append(templist[i] + '_' + ttype + '_' + str(ages[ageidx]))
                    
        print templist[i]
        #Create List of all SN types
        if ttype not in typeList:
            typeList.append(ttype)

                
    return typeList, images, labels, np.array(filenames)

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


def shuffle_arrays(images, labels, filenames):
    imagesShuf = []
    labelsShuf = []
    filenamesShuf = []

    #Randomise order
    indexShuf = range(len(images))
    shuffle(indexShuf)
    for i in indexShuf:
        imagesShuf.append(images[i])
        labelsShuf.append(labels[i])
        filenamesShuf.append((filenames[i]))

    return np.array(imagesShuf), np.array(labelsShuf), np.array(filenamesShuf)

def shortlist_arrays(images, labels, filenames):
    count1a = 0
    count1b = 0
    count1c = 0
    count2 = 0
    imagesShortlist = []
    labelsShortlist = []
    filenamesShortlist = []

    imagesShuf, labelsShuf, filenamesShuf = shuffle_arrays(images, labels, filenames)

    for i in range(len(labelsShuf)):
        label = labelsShuf[i]
        image = imagesShuf[i]
        filename = filenamesShuf[i]
        
        if (label[0] == 1):
            count1a += 1
            if (count1a > 5):
                continue
        elif (label[1] == 1):
            count1b += 1
            if (count1b > 5):
                print("1b exceed")
                continue
        elif (label[2] == 1):
            count1c += 1
            if (count1c > 5):
                print("1c exceed")
                continue
        elif (label[3] == 1):
            count2 += 1
            if (count2 > 5):
                print("2 exceed")
                continue
            
        imagesShortlist.append(image)
        labelsShortlist.append(label)
        filenamesShortlist.append(filename)
        print(count1a, count1b, count1c, count2)


    imagesShortlist = np.array(imagesShortlist)
    labelsShortlist = np.array(labelsShortlist)
    filenamesShortlist = np.array(filenamesShortlist)
    imagesShortlistShuf, labelsShortlistShuf, filenamesShortlistShuf = shuffle_arrays(imagesShortlist, labelsShortlist, filenamesShortlist)
        
    return imagesShortlistShuf, labelsShortlistShuf

"""Randomise order and shortlist arrays to have even
    number of SN for each type """
def over_sample_arrays(images, labels, filenames):
    counts = np.zeros(ntypes)
    imagesShortlist = []
    labelsShortlist = []
    filenamesShortlist = []

    imagesShuf, labelsShuf, filenamesShuf = shuffle_arrays(images, labels, filenames)

    for i in range(len(labelsShuf)):
        label = labelsShuf[i]
        image = imagesShuf[i]
        filename = filenamesShuf[i]
        
        if (label[0] == 1): 
            for r in range(20):
                imagesShortlist.append(image)# + abs(np.random.normal(0,0.1,100)))
                labelsShortlist.append(label)
                filenamesShortlist.append(filename)
                counts = np.array(label) + counts
        elif (label[1] == 1): 
            for r in range(177):
                imagesShortlist.append(image)
                labelsShortlist.append(label)
                filenamesShortlist.append(filename)
                counts = np.array(label) + counts
        elif (label[2] == 1): 
            for r in range(164):
                imagesShortlist.append(image)
                labelsShortlist.append(label)
                filenamesShortlist.append(filename)
                counts = np.array(label) + counts
        elif (label[3] == 1):
            for r in range(177):
                imagesShortlist.append(image)
                labelsShortlist.append(label)
                filenamesShortlist.append(filename)
                counts = np.array(label) + counts
        elif (label[4] == 1): 
            for r in range(287):
                imagesShortlist.append(image)
                labelsShortlist.append(label)
                filenamesShortlist.append(filename)
                counts = np.array(label) + counts
        elif (label[5] == 1): 
            for r in range(88):
                imagesShortlist.append(image)
                labelsShortlist.append(label)
                filenamesShortlist.append(filename)
                counts = np.array(label) + counts
        elif (label[6] == 1):
            for r in range(88):
                imagesShortlist.append(image)
                labelsShortlist.append(label)
                filenamesShortlist.append(filename)
                counts = np.array(label) + counts
        elif (label[7] == 1): 
            for r in range(255):
                imagesShortlist.append(image)
                labelsShortlist.append(label)
                filenamesShortlist.append(filename)
                counts = np.array(label) + counts
        elif (label[8] == 1): 
            for r in range(575):
                imagesShortlist.append(image)
                labelsShortlist.append(label)
                filenamesShortlist.append(filename)
                counts = np.array(label) + counts
        elif (label[9] == 1):
            for r in range(230):
                imagesShortlist.append(image)
                labelsShortlist.append(label)
                filenamesShortlist.append(filename)
                counts = np.array(label) + counts
        elif (label[10] == 1): 
            for r in range(128):
                imagesShortlist.append(image)
                labelsShortlist.append(label)
                filenamesShortlist.append(filename)
                counts = np.array(label) + counts
        elif (label[11] == 1): 
            for r in range(767):
                imagesShortlist.append(image)
                labelsShortlist.append(label)
                filenamesShortlist.append(filename)
                counts = np.array(label) + counts
        elif (label[12] == 1):
            for r in range(288):
                imagesShortlist.append(image)
                labelsShortlist.append(label)
                filenamesShortlist.append(filename)
                counts = np.array(label) + counts
        elif (label[13] == 1): 
            for r in range(153):
                imagesShortlist.append(image)
                labelsShortlist.append(label)
                filenamesShortlist.append(filename)
                counts = np.array(label) + counts


        print(counts)


    imagesShortlist = np.array(imagesShortlist)
    labelsShortlist = np.array(labelsShortlist)
    filenamesShortlist = np.array(filenamesShortlist)
    imagesShortlistShuf, labelsShortlistShuf, filenamesShortlistShuf = shuffle_arrays(imagesShortlist, labelsShortlist, filenamesShortlist)
        
    return imagesShortlistShuf, labelsShortlistShuf, filenamesShortlistShuf
        

def all_templates_to_arrays(snidtempfilelist, sftempfilelist):
    images = np.empty((0,N), np.float32) #Number of pixels
    labels = np.empty((0,ntypes), float) #Number of labels (SN types)
    typeList = []

    typelistSnid, imagesSnid, labelsSnid, filenamesSnid = snid_templates_to_arrays(snidtempfilelist)
    #imagesSuperfit, labelsSuperfit, filenamesSuperfit = superfit_templates_to_arrays(sftempfilelist)
    
    images = np.vstack((imagesSnid))#, imagesSuperfit)) #Add in other templates from superfit etc.
    labels = np.vstack((labelsSnid))#, labelsSuperfit))
    filenames = np.hstack((filenamesSnid))#, filenamesSuperfit))
    
    typeList = typelistSnid
    
    
    imagesShuf, labelsShuf, filenamesShuf = shuffle_arrays(images, labels, filenames)#imagesShortlist, labelsShortlist = shortlist_arrays(images, labels)

    
    return typeList, imagesShuf, labelsShuf, filenamesShuf #imagesShortlist, labelsShortlist


def sort_data(snidtempfilelist, sftempfilelist):
    trainPercentage = 1.
    testPercentage = 0.
    validatePercentage = 0.
    
    typeList, images, labels, filenames = all_templates_to_arrays(snidtempfilelist, sftempfilelist)
    imagesSuperfit, labelsSuperfit, filenamesSuperfit = superfit_templates_to_arrays(sftempfilelist)
    imagesSuperfit, labelsSuperfit, filenamesSuperfit = shuffle_arrays(imagesSuperfit, labelsSuperfit, filenamesSuperfit)


    trainSize = int(trainPercentage*len(images))
    testSize = int(testPercentage*len(images))
    
    trainImages = images[:trainSize]
    testImages = imagesSuperfit#images[trainSize : trainSize+testSize]
    validateImages = images[trainSize+testSize:]
    trainLabels = labels[:trainSize]
    testLabels = labelsSuperfit#labels[trainSize : trainSize+testSize]
    validateLabels = labels[trainSize+testSize:]
    trainFilenames = filenames[:trainSize]
    testFilenames = filenamesSuperfit#filenames[trainSize : trainSize+testSize]
    validateFilenames = filenames[trainSize+testSize:]

    trainImagesOverSample, trainLabelsOverSample, trainFilenamesOverSample = over_sample_arrays(trainImages, trainLabels, trainFilenames)
    testImagesShortlist, testLabelsShortlist, testFilenamesShortlist = testImages, testLabels, testFilenames#shortlist_arrays(testImages, testLabels)

    return ((trainImagesOverSample, trainLabelsOverSample, trainFilenamesOverSample), (testImagesShortlist, testLabelsShortlist, testFilenamesShortlist), (validateImages, validateLabels, validateFilenames))



##a = all_templates_to_arrays(snidtempfilelist1, sftempfilelist1)
##print a[0]
##print ("-------------------------------")
##print a[1]
##print ("-------------------------------")
##print a[2]
##print ("-------------------------------")

sortData = sort_data(snidtempfilelist1, sftempfilelist1)

trainImages = sortData[0][0]
trainLabels = sortData[0][1]
trainFilenames = sortData[0][2]
testImages = sortData[1][0]
testLabels = sortData[1][1]
testFilenames = sortData[1][2]
validateImages = sortData[2][0]
validateLabels = sortData[2][1]
validateFilenames = sortData[2][2]

np.savez_compressed('file1.npz', trainImages=trainImages, trainLabels=trainLabels, trainFilenames=trainFilenames, testImages = testImages, testLabels=testLabels, testFilenames=testFilenames)



    
