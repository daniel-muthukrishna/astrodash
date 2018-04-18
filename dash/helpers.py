import numpy as np


def temp_list(tempFileList):
    f = open(tempFileList, 'rU')

    fileList = f.readlines()
    for i in range(0, len(fileList)):
        fileList[i] = fileList[i].strip('\n')

    f.close()

    return fileList


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0  # -inf inf NaN
    return c