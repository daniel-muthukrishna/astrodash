import os
import glob
import pickle
import shutil
import numpy as np
import itertools
from astrodash.multilayer_convnet import build_model
import zipfile
import gzip
import time
from astrodash.array_tools import labels_indexes_to_arrays
from astrodash.model_metrics import calc_model_metrics
from astrodash.create_arrays import OverSampling
from astrodash.helpers import redshift_binned_spectrum, calc_params_for_log_redshifting


def train_model(dataDirName, overwrite=False, numTrainBatches=500000, minZ=0., maxZ=0., redshifting=False, batch_size=50):
    """  Train model. Unzip and overwrite exisiting training set if overwrite is True"""

    modelFilename = os.path.join(dataDirName, "DASH_keras_model.hdf5")
    if os.path.exists(modelFilename):
        return modelFilename

    # Open training data files
    trainingSet = os.path.join(dataDirName, 'training_set.zip')
    extractedFolder = os.path.join(dataDirName, 'training_set')
    # zipRef = zipfile.ZipFile(trainingSet, 'r')
    # zipRef.extractall(extractedFolder)
    # zipRef.close()
    if os.path.exists(extractedFolder) and overwrite:
        shutil.rmtree(extractedFolder, ignore_errors=True)
        os.system(f'unzip "{trainingSet}" -d "{extractedFolder}"')
    elif not os.path.exists(extractedFolder):
        os.system(f'unzip "{trainingSet}" -d "{extractedFolder}"')
    else:
        pass

    npyFiles = {}
    fileList = os.listdir(extractedFolder)
    for filename in fileList:
        f = os.path.join(extractedFolder, filename)
        if filename.endswith('.gz'):
            # # npyFiles[filename.strip('.npy.gz')] = gzip.GzipFile(f, 'r')
            # gzFile = gzip.open(f, "rb")
            # unCompressedFile = open(f.strip('.gz'), "wb")
            # decoded = gzFile.read()
            # unCompressedFile.write(decoded)
            # gzFile.close()
            # unCompressedFile.close()
            npyFiles[filename.strip('.npy.gz')] = f.strip('.gz')
            os.system(f'gzip -d "{f}"')  # "gzip -dk %s" % f
        elif filename.endswith('.npy'):
            npyFiles[filename.strip('.npy')] = f

    trainImages = np.load(npyFiles['trainImages'], mmap_mode='r')
    trainLabels = np.load(npyFiles['trainLabels'], mmap_mode='r')
    testImages = np.load(npyFiles['testImages'], mmap_mode='r')
    testLabelsIndexes = np.load(npyFiles['testLabels'], mmap_mode='r')
    typeNamesList = np.load(npyFiles['typeNamesList'])
    testTypeNames = np.load(npyFiles['testTypeNames'])
    # testImages = np.load(npyFiles['testImagesNoGal'], mmap_mode='r')
    # testLabelsIndexes = np.load(npyFiles['testLabelsNoGal'], mmap_mode='r')

    print("Completed creatingArrays")
    print(len(trainImages))

    nLabels = len(typeNamesList)
    N = 1024
    nIndexes, dwlog, w0, w1, nw = calc_params_for_log_redshifting(dataDirName)
    with open(os.path.join(dataDirName, 'training_params.pickle'), 'rb') as f1:
        pars = pickle.load(f1)
    snTypes = pars['typeList']

    overSampling = OverSampling(nLabels, N, images=trainImages, labels=trainLabels)
    trainArrays = overSampling.over_sample_arrays(smote=False)
    trainImages, trainLabels = trainArrays['images'], trainArrays['labels']

    testLabels = labels_indexes_to_arrays(testLabelsIndexes, nLabels)
    # testImages = testImages[0:400]

    model = build_model(N, nLabels)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    trainImagesCycle = itertools.cycle(trainImages)
    trainLabelsCycle = itertools.cycle(trainLabels)
    train_acc, test_acc = [], []
    for i in range(numTrainBatches):
        batch_xs = np.array(list(itertools.islice(trainImagesCycle, batch_size * i, batch_size * i + batch_size)))
        batch_ys = labels_indexes_to_arrays(list(itertools.islice(trainLabelsCycle, batch_size * i, batch_size * i + batch_size)), nLabels)

        # Redshift arrays
        if redshifting is True:
            redshifts = np.random.uniform(low=minZ, high=maxZ, size=len(batch_xs))
            for j, z in enumerate(redshifts):
                batch_xs[j] = redshift_binned_spectrum(batch_xs[j], z, nIndexes, dwlog, w0, w1, nw, outerVal=0.5)

        batch_metrics = model.train_on_batch(batch_xs, batch_ys, return_dict=True)
        test_metrics = model.test_on_batch(testImages, testLabels, return_dict=True)
        train_acc.append(batch_metrics['accuracy'])
        test_acc.append(test_metrics['accuracy'])
        print(f"Batch {i} of {numTrainBatches}", f"Training metrics: {batch_metrics}", f"Testing metrics: {test_metrics}")

    # SAVE THE MODEL
    model.save(modelFilename)

    try:
        import matplotlib.pyplot as plt
        num_objects = np.arange(0, numTrainBatches*batch_size, batch_size)
        plt.plot(num_objects, train_acc, label='Training set')
        plt.plot(num_objects, test_acc, label='Testing set')
        plt.xlabel("Train objects")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(dataDirName, "accuracy.png"))
        np.savetxt(os.path.join(dataDirName, "accuracy.txt"), np.array([train_acc, test_acc]))
    except Exception as e:
        print(e)

    # Delete temporary memory mapping files
    for filename in glob.glob('shuffled*.dat') + glob.glob('oversampled*.dat'):
        if not os.path.samefile(filename, trainImages.filename) and not os.path.samefile(filename, trainLabels.filename):
            os.remove(filename)

    # try:
    calc_model_metrics(modelFilename, testLabelsIndexes, testImages, testTypeNames, typeNamesList, snTypes, fig_dir=dataDirName)
    # except Exception as e:
    #     print(e)

    return modelFilename


if __name__ == '__main__':
    t1 = time.time()
    savedFilenames = train_model('data_files_zeroZ/', overwrite=False, numTrainBatches=500000, redshifting=True, minZ=0,
                                 maxZ=0.8)
    t2 = time.time()
    print("time spent: {0:.2f}".format(t2 - t1))
