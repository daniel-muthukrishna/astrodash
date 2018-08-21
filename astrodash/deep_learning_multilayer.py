import os
import glob
import shutil
import numpy as np
import itertools
import tensorflow as tf
from astrodash.multilayer_convnet import convnet_variables
import zipfile
import gzip
import time
from astrodash.array_tools import labels_indexes_to_arrays
from astrodash.model_metrics import calc_model_metrics
from astrodash.create_arrays import OverSampling
from astrodash.helpers import redshift_binned_spectrum, calc_params_for_log_redshifting


def train_model(dataDirName, overwrite=False, numTrainBatches=500000, minZ=0., maxZ=0., redshifting=False):
    """  Train model. Unzip and overwrite exisiting training set if overwrite is True"""
    # Open training data files
    trainingSet = os.path.join(dataDirName, 'training_set.zip')
    extractedFolder = os.path.join(dataDirName, 'training_set')
    # zipRef = zipfile.ZipFile(trainingSet, 'r')
    # zipRef.extractall(extractedFolder)
    # zipRef.close()
    if os.path.exists(extractedFolder) and overwrite:
        shutil.rmtree(extractedFolder, ignore_errors=True)
        os.system("unzip %s -d %s" % (trainingSet, extractedFolder))
    elif not os.path.exists(extractedFolder):
        os.system("unzip %s -d %s" % (trainingSet, extractedFolder))
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
            os.system("gzip -d %s" % f)  # "gzip -dk %s" % f
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

    overSampling = OverSampling(nLabels, N, images=trainImages, labels=trainLabels)
    trainArrays = overSampling.over_sample_arrays(smote=False)
    trainImages, trainLabels = trainArrays['images'], trainArrays['labels']

    # # Delete temporary memory mapping files
    # for filename in glob.glob('shuffled*.dat') + glob.glob('oversampled*.dat'):
    #     if not os.path.samefile(filename, trainImages.filename) and not os.path.samefile(filename, trainLabels.filename):
    #         os.remove(filename)

    # Set up the convolutional network architecture
    imWidth = 32  # Image size and width
    imWidthReduc = 8
    a = []
    x, y_, keep_prob, y_conv, W, b = convnet_variables(imWidth, imWidthReduc, N, nLabels)

    with tf.Session() as sess: # config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)) as sess:
        # TRAIN AND EVALUATE MODEL
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv + 1e-8), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess.run(tf.global_variables_initializer())

        testLabels = labels_indexes_to_arrays(testLabelsIndexes[0:400], nLabels)
        testImages = testImages[0:400]
        # testLabelsWithGal = labels_indexes_to_arrays(testLabelsIndexesWithGal[0:200], nLabels)
        # testImagesWithGal = testImagesWithGal[0:200]

        trainImagesCycle = itertools.cycle(trainImages)
        trainLabelsCycle = itertools.cycle(trainLabels)
        for i in range(numTrainBatches):
            batch_xs = np.array(list(itertools.islice(trainImagesCycle, 50 * i, 50 * i + 50)))
            batch_ys = labels_indexes_to_arrays(list(itertools.islice(trainLabelsCycle, 50 * i, 50 * i + 50)), nLabels)

            # Redshift arrays
            if redshifting is True:
                redshifts = np.random.uniform(low=minZ, high=maxZ, size=len(batch_xs))
                for j, z in enumerate(redshifts):
                    batch_xs[j] = redshift_binned_spectrum(batch_xs[j], z, nIndexes, dwlog, w0, w1, nw, outerVal=0.5)

            train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
                testAcc = accuracy.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0})
                print("test accuracy %g" % testAcc)
                a.append(testAcc)
                # if i % 1000 == 0:
                #     testWithGalAcc = accuracy.eval(feed_dict={x: testImagesWithGal, y_: testLabelsWithGal, keep_prob: 1.0})
                #     print("test With Gal accuracy %g" % testWithGalAcc)

        print("test accuracy %g" % accuracy.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0}))

        yy = y_conv.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0})
        cp = correct_prediction.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0})
        print(cp)
        for i in range(len(cp)):
            if cp[i] == False:
                predictedIndex = np.argmax(yy[i])
                print(i, testTypeNames[i], typeNamesList[predictedIndex])

        # SAVE THE MODEL
        saveFilename = os.path.join(dataDirName, "tensorflow_model.ckpt")
        saver = tf.train.Saver()
        save_path = saver.save(sess, saveFilename)
        print("Model saved in file: %s" % save_path)

    modelFilenames = [saveFilename + '.index', saveFilename + '.meta', saveFilename + '.data-00000-of-00001']

    try:
        import matplotlib.pyplot as plt
        plt.plot(a)
        plt.xlabel("Number of Epochs")
        plt.ylabel("Testing accuracy")
        plt.savefig(os.path.join(dataDirName, "testing_accuracy.png"))
        np.savetxt(os.path.join(dataDirName, "testing_accuracy.txt"), np.array(a))
    except Exception as e:
        print(e)

    try:
        calc_model_metrics(saveFilename, testLabelsIndexes, testImages, testTypeNames, typeNamesList)
    except Exception as e:
        print(e)

    return modelFilenames


if __name__ == '__main__':
    t1 = time.time()
    savedFilenames = train_model('data_files_zeroZ/', overwrite=False, numTrainBatches=500000, redshifting=True, minZ=0, maxZ=0.8)
    t2 = time.time()
    print("time spent: {0:.2f}".format(t2 - t1))

