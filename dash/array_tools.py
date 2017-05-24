import numpy as np


def labels_indexes_to_arrays(labelsIndexes, nLabels):
    numArrays = len(labelsIndexes)
    labelsIndexes = labelsIndexes
    labels = np.zeros((numArrays, nLabels))
    labels[np.arange(numArrays), labelsIndexes] = 1

    return labels


def zero_non_overlap_part(array, minIndex, maxIndex):
    slicedArray = np.copy(array)
    slicedArray[0:minIndex] = np.zeros(minIndex)
    slicedArray[maxIndex:] = np.zeros(len(array)-maxIndex)

    return slicedArray


def normalise_spectrum(flux):
    fluxNorm = (flux - min(flux)) / (max(flux) - min(flux))

    return fluxNorm