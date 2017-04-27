import numpy as np


def labels_indexes_to_arrays(labelsIndexes, nLabels):
    numArrays = len(labelsIndexes)
    labelsIndexes = labelsIndexes
    labels = np.zeros((numArrays, nLabels))
    labels[np.arange(numArrays), labelsIndexes] = 1

    return labels
