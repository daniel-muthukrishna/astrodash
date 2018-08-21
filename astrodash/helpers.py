import os
import pickle
import numpy as np
import scipy.interpolate
from astrodash.preprocessing import PreProcessSpectrum


def calc_params_for_log_redshifting(dataDirName):
    with open(os.path.join(dataDirName, 'training_params.pickle'), 'rb') as f1:
        pars = pickle.load(f1)
    w0, w1, nw = pars['w0'], pars['w1'], pars['nw']
    n = np.arange(0, int(nw))

    dwlog = np.log(w1 / w0) / nw
    # wlog = w0 * np.exp(n * dwlog)
    # waveInterp = scipy.interpolate.interp1d(n, wlog)  # Will interpolate any value within wavelength given an index

    return n, dwlog, w0, w1, nw


def redshift_binned_spectrum(flux, z, nIndexes, dwlog, w0, w1, nw, outerVal=0.5):
    # assert len(flux) == nw
    redshiftedIndexes = nIndexes + np.log(1 + z) / dwlog
    indexesInRange = redshiftedIndexes[redshiftedIndexes > 0]
    fluxInterp = scipy.interpolate.interp1d(indexesInRange, flux[redshiftedIndexes > 0], kind='linear')

    minWaveIndex = int(indexesInRange[0])

    fluxRedshifted = np.zeros(nw)
    fluxRedshifted[0:minWaveIndex] = outerVal * np.ones(minWaveIndex)
    fluxRedshifted[minWaveIndex:] = fluxInterp(indexesInRange)[:nw-minWaveIndex]

    # Apodize edges
    preprocess = PreProcessSpectrum(w0, w1, nw)
    minIndex, maxIndex = preprocess.processingTools.min_max_index(fluxRedshifted, outerVal=outerVal)
    apodizedFlux = preprocess.apodize(fluxRedshifted, minIndex, maxIndex, outerVal=outerVal)

    return apodizedFlux


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
