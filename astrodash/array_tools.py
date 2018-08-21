import numpy as np


def labels_indexes_to_arrays(labelsIndexes, nLabels):
    numArrays = len(labelsIndexes)
    labelsIndexes = labelsIndexes
    labels = np.zeros((numArrays, nLabels))
    labels[np.arange(numArrays), labelsIndexes] = 1

    return labels


def zero_non_overlap_part(array, minIndex, maxIndex, outerVal=0.):
    slicedArray = np.copy(array)
    slicedArray[0:minIndex] = outerVal * np.ones(minIndex)
    slicedArray[maxIndex:] = outerVal * np.ones(len(array)-maxIndex)

    return slicedArray


def normalise_spectrum(flux):
    if len(flux) == 0 or min(flux) == max(flux):  # No data
        fluxNorm = np.zeros(len(flux))
    else:
        fluxNorm = (flux - min(flux)) / (max(flux) - min(flux))

    return fluxNorm


def mean_zero_spectra(flux, minIndex, maxIndex, nw):
    fluxOut = np.copy(flux)
    fluxOut[0:minIndex] = np.full(minIndex, fluxOut[minIndex])
    fluxOut[maxIndex:] = np.full(nw - maxIndex, fluxOut[minIndex])
    fluxOut = fluxOut - fluxOut[minIndex]

    return fluxOut
