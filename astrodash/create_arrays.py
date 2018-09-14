import numpy as np
from random import shuffle
import multiprocessing as mp
import itertools

from astrodash.helpers import temp_list, div0
from astrodash.sn_processing import PreProcessing
from astrodash.combine_sn_and_host import training_template_data
from astrodash.preprocessing import ProcessingTools
from astrodash.array_tools import zero_non_overlap_part, normalise_spectrum

try:
    from imblearn import over_sampling
    IMBLEARN_EXISTS = True
except ImportError:
    IMBLEARN_EXISTS = False


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

            if ageBin != ageBinPrev:
                ageLabelMax = int(round(age))
                ageLabels.append(str(int(ageLabelMin)) + " to " + str(ageLabelMax))
                ageLabelMin = ageLabelMax

            ageBinPrev = ageBin

        ageLabels.append(str(int(ageLabelMin)) + " to " + str(int(self.maxAge)))

        return ageLabels


class CreateLabels(object):

    def __init__(self, nTypes, minAge, maxAge, ageBinSize, typeList, hostList, nHostTypes):
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize
        self.typeList = typeList
        self.ageBinning = AgeBinning(self.minAge, self.maxAge, self.ageBinSize)
        self.numOfAgeBins = self.ageBinning.age_bin(self.maxAge-0.1) + 1
        self.nLabels = self.nTypes * self.numOfAgeBins
        self.ageLabels = self.ageBinning.age_labels()
        self.hostList = hostList
        self.nHostTypes = nHostTypes

    def label_array(self, ttype, age, host=None):
        ageBin = self.ageBinning.age_bin(age)

        try:
            typeIndex = self.typeList.index(ttype)
        except ValueError as err:
            raise Exception("INVALID TYPE: {0}".format(err))

        if host is None:
            labelArray = np.zeros((self.nTypes, self.numOfAgeBins))
            labelArray[typeIndex][ageBin] = 1
            labelArray = labelArray.flatten()
            typeName = ttype + ": " + self.ageLabels[ageBin]
        else:
            hostIndex = self.hostList.index(host)
            labelArray = np.zeros((self.nHostTypes, self.nTypes, self.numOfAgeBins))
            labelArray[hostIndex][typeIndex][ageBin] = 1
            labelArray = labelArray.flatten()
            typeName = "{}: {}: {}".format(host, ttype, self.ageLabels[ageBin])

        labelIndex = np.argmax(labelArray)

        return labelIndex, typeName

    def type_names_list(self):
        typeNamesList = []
        if self.hostList is None:
            for tType in self.typeList:
                for ageLabel in self.ageBinning.age_labels():
                    typeNamesList.append("{}: {}".format(tType, ageLabel))
        else:
            for host in self.hostList:
                for tType in self.typeList:
                    for ageLabel in self.ageBinning.age_labels():
                        typeNamesList.append("{}: {}: {}".format(host, tType, ageLabel))

        return np.array(typeNamesList)


class ReadSpectra(object):

    def __init__(self, w0, w1, nw, snFilename, galFilename=None):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.snFilename = snFilename
        self.galFilename = galFilename
        if galFilename is None:
            self.data = PreProcessing(snFilename, w0, w1, nw)

    def sn_plus_gal_template(self, snAgeIdx, snCoeff, galCoeff, z):
        wave, flux, minIndex, maxIndex, nCols, ages, tType = training_template_data(snAgeIdx, snCoeff, galCoeff, z, self.snFilename, self.galFilename, self.w0, self.w1, self.nw)

        return wave, flux, nCols, ages, tType, minIndex, maxIndex

    def input_spectrum(self, z, smooth, minWave, maxWave):
        wave, flux, minIndex, maxIndex, z = self.data.two_column_data(z, smooth, minWave, maxWave)

        return wave, flux, int(minIndex), int(maxIndex), z


class ArrayTools(object):

    def __init__(self, nLabels, nw):
        self.nLabels = nLabels
        self.nw = nw

    def shuffle_arrays(self, memmapName='', **kwargs):
        """ Must take images and labels as arguments with the keyword specified.
        Can optionally take filenames and typeNames as arguments """
        arraySize = len(kwargs['labels'])
        if arraySize == 0:
            return kwargs

        kwargShuf = {}
        self.randnum = np.random.randint(10000)
        for key in kwargs:
            if key == 'images':
                arrayShuf = np.memmap('shuffled_{}_{}_{}.dat'.format(key, memmapName, self.randnum), dtype=np.float16, mode='w+', shape=(arraySize, int(self.nw)))
            elif key == 'labels':
                arrayShuf = np.memmap('shuffled_{}_{}_{}.dat'.format(key, memmapName, self.randnum), dtype=np.uint16, mode='w+', shape=arraySize)
            else:
                arrayShuf = np.memmap('shuffled_{}_{}_{}.dat'.format(key, memmapName, self.randnum), dtype=object, mode='w+', shape=arraySize)
            kwargShuf[key] = arrayShuf

        print("Shuffling...")
        # Randomise order
        p = np.random.permutation(len(kwargs['labels']))
        for key in kwargs:
            assert len(kwargs[key]) == arraySize
            print(key, "shuffling...")
            print(len(p))
            kwargShuf[key] = kwargs[key][p]

        return kwargShuf

    def count_labels(self, labels):
        counts = np.zeros(self.nLabels)

        for i in range(len(labels)):
            counts[labels[i]] += 1

        return counts

    def augment_data(self, flux, stdDevMean=0.05, stdDevStdDev=0.05):
        minIndex, maxIndex = ProcessingTools().min_max_index(flux, outerVal=0.5)
        noise = np.zeros(self.nw)
        stdDev = abs(np.random.normal(stdDevMean, stdDevStdDev)) # randomised standard deviation
        noise[minIndex:maxIndex] = np.random.normal(0, stdDev, maxIndex - minIndex)
        # # Add white noise to regions outside minIndex to maxIndex
        # noise[0:minIndex] = np.random.uniform(0.0, 1.0, minIndex)
        # noise[maxIndex:] = np.random.uniform(0.0, 1.0, self.nw-maxIndex)

        augmentedFlux = flux + noise
        augmentedFlux = normalise_spectrum(augmentedFlux)
        augmentedFlux = zero_non_overlap_part(augmentedFlux, minIndex, maxIndex, outerVal=0.5)

        return augmentedFlux


class OverSampling(ArrayTools):
    def __init__(self, nLabels, nw, **kwargs):
        """ Must take images and labels as arguments with the keyword specified.
        Can optionally take filenames and typeNames as arguments """
        ArrayTools.__init__(self, nLabels, nw)
        self.kwargs = kwargs

        counts = self.count_labels(self.kwargs['labels'])
        print("Before OverSample")  #
        print(counts)  #

        self.overSampleAmount = np.rint(div0(1 * max(counts), counts))  # ignore zeros in counts
        self.overSampleArraySize = int(sum(np.array(self.overSampleAmount, int) * counts))
        print(np.array(self.overSampleAmount, int) * counts)
        print(np.array(self.overSampleAmount, int))
        print(self.overSampleArraySize, len(self.kwargs['labels']))
        self.kwargOverSampled = {}
        self.randnum = np.random.randint(10000)
        for key in self.kwargs:
            if key == 'images':
                arrayOverSampled = np.memmap('oversampled_{}_{}.dat'.format(key, self.randnum), dtype=np.float16, mode='w+',
                                             shape=(self.overSampleArraySize, int(self.nw)))
            elif key == 'labels':
                arrayOverSampled = np.memmap('oversampled_{}_{}.dat'.format(key, self.randnum), dtype=np.uint16, mode='w+',
                                             shape=self.overSampleArraySize)
            else:
                arrayOverSampled = np.memmap('oversampled_{}_{}.dat'.format(key, self.randnum), dtype=object, mode='w+',
                                             shape=self.overSampleArraySize)
            self.kwargOverSampled[key] = arrayOverSampled

        self.kwargShuf = self.shuffle_arrays(memmapName='pre-oversample_{}'.format(self.randnum), **self.kwargs)
        print(len(self.kwargShuf['labels']))

    def oversample_mp(self, i_in, offset_in, std_in, labelIndex_in):
        print('oversampling', i_in, len(self.kwargShuf['labels']))
        oversampled = {key: [] for key in self.kwargs}
        repeatAmount = int(self.overSampleAmount[labelIndex_in])
        for r in range(repeatAmount):
            for key in self.kwargs:
                if key == 'images':
                    oversampled[key].append(self.augment_data(self.kwargShuf[key][i_in], stdDevMean=0.05, stdDevStdDev=std_in))
                else:
                    oversampled[key].append(self.kwargShuf[key][i_in])
        return oversampled, offset_in, repeatAmount

    def collect_results(self, result):
        """Uses apply_async's callback to setup up a separate Queue for each process"""
        oversampled_in, offset_in, repeatAmount = result
        for key in self.kwargs:
            rlength_array = np.array(oversampled_in[key])
            self.kwargOverSampled[key][offset_in:repeatAmount+offset_in] = rlength_array[:]

    def over_sample_arrays(self, smote=False):
        if smote:
            return self.smote_oversample()
        else:
            return self.minority_oversample_with_noise()

    def minority_oversample_with_noise(self):
        offset = 0
        # pool = mp.Pool()
        for i in range(len(self.kwargShuf['labels'])):
            labelIndex = self.kwargShuf['labels'][i]
            if self.overSampleAmount[labelIndex] < 10:
                std = 0.03
            else:
                std = 0.05
            # pool.apply_async(self.oversample_mp, args=(i, offset, std, labelIndex), callback=self.collect_results)
            self.collect_results(self.oversample_mp(i, offset, std, labelIndex))
            offset += int(self.overSampleAmount[labelIndex])
        # pool.close()
        # pool.join()

        # for i, output in enumerate(outputs):
        #     self.collect_results(output)
        #     print('combining results...', i, len(outputs))

        print("Before Shuffling")
        self.kwargOverSampledShuf = self.shuffle_arrays(memmapName='oversampled_{}'.format(self.randnum), **self.kwargOverSampled)
        print("After Shuffling")

        return self.kwargOverSampledShuf

    def smote_oversample(self):
        sm = over_sampling.SMOTE(random_state=42, n_jobs=30)
        images, labels = sm.fit_sample(X=self.kwargShuf['images'], y=self.kwargShuf['labels'])

        self.kwargOverSampledShuf = self.shuffle_arrays(memmapName='oversampled_smote_{}'.format(self.randnum), images=images, labels=labels)

        return self.kwargOverSampledShuf


class CreateArrays(object):
    def __init__(self, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, numOfRedshifts, hostTypes=None, nHostTypes=None):
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
        self.numOfRedshifts = numOfRedshifts
        self.ageBinning = AgeBinning(minAge, maxAge, ageBinSize)
        self.numOfAgeBins = self.ageBinning.age_bin(maxAge-0.1) + 1
        self.nLabels = nTypes * self.numOfAgeBins * nHostTypes
        self.createLabels = CreateLabels(self.nTypes, self.minAge, self.maxAge, self.ageBinSize, self.typeList, hostTypes, nHostTypes)
        self.hostTypes = hostTypes

    def combined_sn_gal_templates_to_arrays(self, args):
        snTemplateLocation, snTempList, galTemplateLocation, galTempList, snFractions, ageIndexes = args
        images = np.empty((0, int(self.nw)), np.float16)  # Number of pixels
        labelsIndexes = []
        filenames = []
        typeNames = []

        for j, gal in enumerate(galTempList):
            galFilename = galTemplateLocation + gal if galTemplateLocation is not None else None
            for i, sn in enumerate(snTempList):
                nCols = 15
                readSpectra = ReadSpectra(self.w0, self.w1, self.nw, snTemplateLocation + sn, galFilename)
                for ageidx in ageIndexes[sn]:
                    if ageidx >= nCols:
                        break
                    for snCoeff in snFractions:
                        galCoeff = 1 - snCoeff
                        if self.numOfRedshifts == 1:
                            redshifts = [self.minZ]
                        else:
                            redshifts = np.random.uniform(low=self.minZ, high=self.maxZ, size=self.numOfRedshifts)
                        for z in redshifts:
                            tempWave, tempFlux, nCols, ages, tType, tMinIndex, tMaxIndex = readSpectra.sn_plus_gal_template(ageidx, snCoeff, galCoeff, z)
                            if tMinIndex == tMaxIndex or not tempFlux.any():
                                print("NO DATA for {} {} ageIdx:{} z>={}".format(galTempList[j], snTempList[i], ageidx, z))
                                break

                            if self.minAge < float(ages[ageidx]) < self.maxAge:
                                if self.hostTypes is None:  # Checks if we are classifying by host as well
                                    labelIndex, typeName = self.createLabels.label_array(tType, ages[ageidx], host=None)
                                else:
                                    labelIndex, typeName = self.createLabels.label_array(tType, ages[ageidx], host=galTempList[j])
                                if tMinIndex > (self.nw - 1):
                                    continue
                                nonzeroflux = tempFlux[tMinIndex:tMaxIndex + 1]
                                newflux = (nonzeroflux - min(nonzeroflux)) / (max(nonzeroflux) - min(nonzeroflux))
                                newflux2 = np.concatenate((tempFlux[0:tMinIndex], newflux, tempFlux[tMaxIndex + 1:]))
                                images = np.append(images, np.array([newflux2]), axis=0)
                                labelsIndexes.append(labelIndex) # labels = np.append(labels, np.array([label]), axis=0)
                                filenames.append("{0}_{1}_{2}_{3}_snCoeff{4}_z{5}".format(snTempList[i], tType, str(ages[ageidx]), galTempList[j], snCoeff, (z)))
                                typeNames.append(typeName)
                print(snTempList[i], nCols, galTempList[j])

        return images, np.array(labelsIndexes).astype(int), np.array(filenames), np.array(typeNames)

    def collect_results(self, result):
        """Uses apply_async's callback to setup up a separate Queue for each process"""
        imagesPart, labelsPart, filenamesPart, typeNamesPart = result
        self.images.extend(imagesPart)
        self.labelsIndexes.extend(labelsPart)
        self.filenames.extend(filenamesPart)
        self.typeNames.extend(typeNamesPart)

    def combined_sn_gal_arrays_multiprocessing(self, snTemplateLocation, snTempFileList, galTemplateLocation, galTempFileList):
        # TODO: Maybe do memory mapping for these arrays
        self.images = []
        self.labelsIndexes = []
        self.filenames = []
        self.typeNames = []

        if galTemplateLocation is None or galTempFileList is None:
            galTempList = [None]
            galTemplateLocation = None
            snFractions = [1.0]
        else:
            galTempList = temp_list(galTempFileList)
            snFractions = [0.99, 0.98, 0.95, 0.93, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        if isinstance(snTempFileList, dict):
            snTempList = list(snTempFileList.keys())
            ageIndexesDict = snTempFileList
        else:
            snTempList = temp_list(snTempFileList)
            ageIndexesDict = None

        galAndSnTemps = list(itertools.product(galTempList, snTempList))
        argsList = []
        for gal, sn in galAndSnTemps:
            if ageIndexesDict is not None:
                ageIdxDict = {k: ageIndexesDict[k] for k in (sn,)}
            else:
                ageIdxDict = {k: range(0, 1000) for k in (sn,)}
            argsList.append((snTemplateLocation, [sn], galTemplateLocation, [gal], snFractions, ageIdxDict))

        pool = mp.Pool()
        results = pool.map_async(self.combined_sn_gal_templates_to_arrays, argsList)
        pool.close()
        pool.join()

        outputs = results.get()
        for i, output in enumerate(outputs):
            self.collect_results(output)
            print('combining results...', i, len(outputs))

        self.images = np.array(self.images)
        self.labelsIndexes = np.array(self.labelsIndexes)
        self.filenames = np.array(self.filenames)
        self.typeNames = np.array(self.typeNames)

        print("Completed Creating Arrays!")

        return self.images, self.labelsIndexes.astype(np.uint16), self.filenames, self.typeNames
