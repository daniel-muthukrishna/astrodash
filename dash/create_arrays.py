import numpy as np
from random import shuffle
from sn_processing import PreProcessing


class AgeBinning(object):
    def __init__(self, minAge, maxAge, ageBinSize):
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize

    def age_bin(self, age):
        ageBin = int(round(age / self.ageBinSize)) - int(round(self.minAge / self.ageBinSize))

        return ageBin

    def age_labels(self):
        ageLabels = []

        ageBinPrev = 0
        ageLabelMin = self.minAge
        for age in np.arange(self.minAge, self.maxAge, 0.5):
            ageBin = self.age_bin(age)

            if (ageBin != ageBinPrev):
                ageLabelMax = int(round(age))
                ageLabels.append(str(ageLabelMin) + " to " + str(ageLabelMax))
                ageLabelMin = ageLabelMax

            ageBinPrev = ageBin

        ageLabels.append(str(ageLabelMin) + " to " + str(self.maxAge))

        return ageLabels


class CreateLabels(object):

    def __init__(self, nTypes, minAge, maxAge, ageBinSize, typeList):
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize
        self.typeList = typeList
        self.ageBinning = AgeBinning(self.minAge, self.maxAge, self.ageBinSize)
        self.numOfAgeBins = self.ageBinning.age_bin(self.maxAge-0.1) + 1
        self.nLabels = self.nTypes * self.numOfAgeBins
        self.ageLabels = self.ageBinning.age_labels()       
        

    def label_array(self, ttype, age):
        ageBin = self.ageBinning.age_bin(age)
        labelarray = np.zeros((self.nTypes, self.numOfAgeBins))
        typeNames = []

        try:
            typeIndex = self.typeList.index(ttype)
        except ValueError as err:
            print("INVALID TYPE: {0}".format(err))

            
        labelarray[typeIndex][ageBin] = 1
        labelarray = labelarray.flatten()

        typeNames.append(ttype + ": " + self.ageLabels[ageBin])
        typeNames = np.array(typeNames)

        return labelarray, typeNames


    def type_names_list(self):
        typeNamesList = []
        for tType in self.typeList:
            for ageLabel in self.ageBinning.age_labels():
                typeNamesList.append(tType + ": " + ageLabel)

        return typeNamesList
        

class TempList():
    def temp_list(self, tempFileList):
        f = open(tempFileList, 'rU')

        fileList = f.readlines()
        for i in range(0,len(fileList)):
            fileList[i] = fileList[i].strip('\n')

        f.close()

        return fileList
    

class ReadSpectra(object):

    def __init__(self, w0, w1, nw, filename):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.filename = filename
        self.data = PreProcessing(filename, self.w0, self.w1, self.nw)


    def snid_template_data(self, ageIdx, z):
        """ lnw template files """
        wave, flux, nCols, ages, tType, minIndex, maxIndex = self.data.snid_template_data(ageIdx, z)

        return wave, flux, nCols, ages, tType, minIndex, maxIndex


    def sf_age(self):
        snName, extension = self.filename.strip('.dat').split('.')
        ttype, snName = snName.split('/')

        if (extension == 'max'):
            age = 0
        elif (extension[0] == 'p'):
            age = float(extension[1:])
        elif (extension[0] == 'm'):
            age = -float(extension[1:])
        else:
            print "Invalid Superfit Filename: " + self.filename

        return snName, ttype, age


    def superfit_template_data(self, z):
        """ Returns wavelength and flux after all preprocessing """
        wave, flux, minIndex, maxIndex = self.data.two_column_data(z)
        snName, ttype, age = self.sf_age()

        print snName, ttype, age

        return wave, flux, minIndex, maxIndex, age, snName, ttype


    def input_spectrum(self, z, smooth):
        wave, flux, minIndex, maxIndex = self.data.two_column_data(z, smooth)

        return wave, flux, minIndex, maxIndex

class ArrayTools(object):

    def __init__(self, nLabels, nw):
        self.nLabels = nLabels
        self.nw = nw

    def shuffle_arrays(self, images, labels, filenames, typeNames):
        arraySize = len(labels)
        imagesShuf = np.empty((arraySize, int(self.nw)), np.float16)
        labelsShuf = np.empty((arraySize, self.nLabels), np.uint8)
        filenamesShuf = np.empty(arraySize, dtype=object)
        typeNamesShuf = np.empty(arraySize, dtype=object)
        idx = 0
        print("Shuffle2")
        # Randomise order
        indexShuf = range(arraySize)
        shuffle(indexShuf)
        for i in indexShuf:
            imagesShuf[idx] = images[i]
            labelsShuf[idx] = labels[i]
            filenamesShuf[idx] = filenames[i]
            typeNamesShuf[idx] = typeNames[i]
            idx += 1
        print ("LenLabels")
        print(len(labels), idx)
            
        print(imagesShuf)
        print(typeNamesShuf)
        print("Shuffle3")    
        return imagesShuf, labelsShuf, filenamesShuf, typeNamesShuf


    def count_labels(self, labels):
        counts = np.zeros(self.nLabels)

        for i in range(len(labels)):
            counts = labels[i] + counts

        return counts


    def div0(self, a, b):
        """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[~ np.isfinite(c)] = 0  # -inf inf NaN
        return c


    def over_sample_arrays(self, images, labels, filenames, typeNames):
        counts = self.count_labels(labels)
        idx = 0
        print "Before OverSample"  #
        print counts  #


        overSampleAmount = self.div0(1 * max(counts), counts)  # ignore zeros in counts
        overSampleArraySize = int(sum(np.array(overSampleAmount, int) * counts))
        print(np.array(overSampleAmount, int) * counts)
        print(np.array(overSampleAmount, int))
        print(overSampleArraySize, len(labels))
        imagesOverSampled = np.zeros((overSampleArraySize, int(self.nw)), np.float16)
        labelsOverSampled = np.zeros((overSampleArraySize, self.nLabels), np.uint8)
        filenamesOverSampled = np.empty(overSampleArraySize, dtype=object)
        typeNamesOverSampled = np.empty(overSampleArraySize, dtype=object)
        

        counts1 = np.zeros(self.nLabels)

        imagesShuf, labelsShuf, filenamesShuf, typeNamesShuf = self.shuffle_arrays(images, labels, filenames, typeNames)
        print(len(labelsShuf))
        for i in range(len(labelsShuf)):
            label = labelsShuf[i]
            image = imagesShuf[i]
            filename = filenamesShuf[i]
            typeName = typeNamesShuf[i]

            labelIndex = np.argmax(label)
            
            print(idx, i, int(overSampleAmount[labelIndex]))
            for r in range(int(overSampleAmount[labelIndex])):
                imagesOverSampled[idx] = image #np.concatenate(imagesOverSampled, np.array([image]), axis=0)
                labelsOverSampled[idx] = label #np.concatenate(labelsOverSampled, np.array([label]), axis=0)
                filenamesOverSampled[idx] = filename #filenamesOverSampled = np.append(filenamesOverSampled, filename)
                typeNamesOverSampled[idx] = typeName #typeNamesOverSampled = np.append(typeNamesOverSampled, typeName)
                counts1 = label + counts1
                idx += 1
            

                
        print "After OverSample"  #
        print counts1  #

        print("Before Shuffling")
        imagesOverSampledShuf, labelsOverSampledShuf, filenamesOverSampledShuf, typeNamesOverSampledShuf = self.shuffle_arrays(imagesOverSampled, labelsOverSampled, filenamesOverSampled, typeNamesOverSampled)
        print("After Shuffling")
        return imagesOverSampledShuf, labelsOverSampledShuf, filenamesOverSampledShuf, typeNamesOverSampledShuf


class CreateArrays(object):
    def __init__(self, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize
        self.typeList = typeList
        self.minZ = minZ
        self.maxZ = maxZ
        self.redshiftPrecision = 50
        self.numOfRedshifts = (self.maxZ - self.minZ) * self.redshiftPrecision
        self.ageBinning = AgeBinning(self.minAge, self.maxAge, self.ageBinSize)
        self.numOfAgeBins = self.ageBinning.age_bin(self.maxAge-0.1) + 1
        self.nLabels = self.nTypes * self.numOfAgeBins
        self.createLabels = CreateLabels(self.nTypes, self.minAge, self.maxAge, self.ageBinSize, self.typeList)

    def snid_templates_to_arrays(self, snidTemplateLocation, tempfilelist):
        ''' This function is for the SNID processed files, which
            have been preprocessed to negatives, and so cannot be
            imaged yet '''

        templist = TempList().temp_list(tempfilelist) #Arbrirary redshift to read filelist
        typeList = []
        images = np.empty((0, int(self.nw)), np.float32)  # Number of pixels
        labels = np.empty((0, self.nLabels), np.uint8)  # Number of labels (SN types)
        filenames = []#np.empty(0)
        typeNames = []#np.empty(0)
        agesList = []

        for i in range(0, len(templist)):
            ncols = 15
            readSpectra = ReadSpectra(self.w0, self.w1, self.nw, snidTemplateLocation + templist[i])
            
            for ageidx in range(0, 100):
                if (ageidx < ncols):
                    for z in np.linspace(self.minZ, self.maxZ, self.numOfRedshifts + 1):
                        tempwave, tempflux, ncols, ages, ttype, tminindex, tmaxindex = readSpectra.snid_template_data(ageidx, z)
                        agesList.append(ages[ageidx])
                        
                    
                        if ((float(ages[ageidx]) > self.minAge and float(ages[ageidx]) < self.maxAge)):
                            label, typeName = self.createLabels.label_array(ttype, ages[ageidx])
                            nonzeroflux = tempflux[tminindex:tmaxindex + 1]
                            newflux = (nonzeroflux - min(nonzeroflux)) / (max(nonzeroflux) - min(nonzeroflux))
                            newflux2 = np.concatenate((tempflux[0:tminindex], newflux, tempflux[tmaxindex + 1:]))
                            images = np.append(images, np.array([newflux2]), axis=0)  # images.append(newflux2)
                            labels = np.append(labels, np.array([label]), axis=0)  # labels.append(ttype)
                            filenames.append(templist[i] + '_' + ttype + '_' + str(ages[ageidx]) + '_z' + str(z))
                            typeNames.append(typeName)

            print templist[i]
            # Create List of all SN types
            if ttype not in typeList:
                typeList.append(ttype)

        return typeList, images, labels, np.array(filenames), np.array(typeNames)

    def superfit_templates_to_arrays(self, sfTemplateLocation, sftempfilelist):
        templist = TempList().temp_list(sftempfilelist)
        images = np.empty((0, self.nw), np.float32)  # Number of pixels
        labels = np.empty((0, self.nLabels), np.float32)  # Number of labels (SN types)
        filenames = []
        typeNames = []

        for i in range(0, len(templist)):
            readSpectra = ReadSpectra(self.w0, self.w1, self.nw, sfTemplateLocation + templist[i])
            tempwave, tempflux, tminindex, tmaxindex, age, snName, ttype = self.readSpectra.superfit_template_data(z)

            if ((float(ages[ageidx]) > minAge and float(ages[ageidx]) > maxAge)):
                label, typeName = label_array(ttype, ages[ageidx])
                nonzeroflux = tempflux[tminindex:tmaxindex + 1]
                newflux = (nonzeroflux - min(nonzeroflux)) / (max(nonzeroflux) - min(nonzeroflux))
                newflux2 = np.concatenate((tempflux[0:tminindex], newflux, tempflux[tmaxindex + 1:]))
                images = np.append(images, np.array([newflux2]), axis=0)  # images.append(newflux2)
                labels = np.append(labels, np.array([label]), axis=0)  # labels.append(ttype)
                filenames.append(templist[i])
                typeNames.append(typeName)

        return images, labels, np.array(filenames), typeNames


