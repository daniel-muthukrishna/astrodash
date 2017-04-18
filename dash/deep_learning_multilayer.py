import os
import numpy as np
import itertools
import tensorflow as tf
from dash.multilayer_convnet import convnet_variables
import zipfile
import gzip
import time


def train_model(randint=0):
    # Multiprocessing Random Seed
    # print("Set new seed:", randint)
    # np.random.seed(randint)
    # tf.set_random_seed(randint)

    # Open training data files
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    trainingSet = 'data_files/trainingSet_type_age_atRedshiftZero.zip'
    extractedFolder = 'data_files/trainingSet_type_age_atRedshiftZero'
    zipRef = zipfile.ZipFile(trainingSet, 'r')
    zipRef.extractall(extractedFolder)
    zipRef.close()

    npyFiles = {}
    fileList = os.listdir(extractedFolder)
    for filename in fileList:
        if filename.endswith('.gz'):
            f = os.path.join(scriptDirectory, extractedFolder, filename)
            # npyFiles[filename.strip('.npy.gz')] = gzip.GzipFile(f, 'r')
            gzFile = gzip.open(f, "rb")
            unCompressedFile = open(f.strip('.gz'), "wb")
            decoded = gzFile.read()
            unCompressedFile.write(decoded)
            gzFile.close()
            unCompressedFile.close()
            npyFiles[filename.strip('.npy.gz')] = f.strip('.gz')

    trainImages = np.load(npyFiles['trainImages'], mmap_mode='r')
    trainLabels = np.load(npyFiles['trainLabels'], mmap_mode='r')
    testImagesWithGal = np.load(npyFiles['testImages'], mmap_mode='r')
    testLabelsWithGal = np.load(npyFiles['testLabels'], mmap_mode='r')
    typeNamesList = np.load(npyFiles['typeNamesList'])
    testImages = np.load(npyFiles['testImagesNoGal'], mmap_mode='r')
    testLabels = np.load(npyFiles['testLabelsNoGal'], mmap_mode='r')
    testTypeNames = np.load(npyFiles['testTypeNamesNoGal'])

    print("Completed creatingArrays")

    # Set up the convolutional network architecture
    N = 1024
    ntypes = len(testLabels[0])
    imWidth = 32  # Image size and width
    imWidthReduc = 8
    a = []
    x, y_, keep_prob, y_conv = convnet_variables(imWidth, imWidthReduc, N, ntypes)

    with tf.Session() as sess: # config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)) as sess:
        # TRAIN AND EVALUATE MODEL
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv + 1e-8), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess.run(tf.initialize_all_variables())

        trainImagesCycle = itertools.cycle(trainImages)
        trainLabelsCycle = itertools.cycle(trainLabels)
        for i in range(37000):
            batch_xs = np.array(list(itertools.islice(trainImagesCycle, 50 * i, 50 * i + 50)))
            batch_ys = np.array(list(itertools.islice(trainLabelsCycle, 50 * i, 50 * i + 50)))
            train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                print("step %d_%d, training accuracy %g" % (i, randint,train_accuracy))
                testacc = accuracy.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0})
                print("test accuracy %g" % testacc)
                a.append(testacc)
                if i % 1000 == 0:
                    testWithGalacc = accuracy.eval(feed_dict={x: testImagesWithGal[0:200], y_: testLabelsWithGal[0:200], keep_prob: 1.0})
                    print("test With Gal accuracy %g" % testWithGalacc)

        print("test accuracy %g" % accuracy.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0}))

        yy = y_conv.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0})
        cp = correct_prediction.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0})
        print(cp)
        for i in range(len(cp)):
            if cp[i] == False:
                predictedIndex = np.argmax(yy[i])
                print(i, testTypeNames[i], typeNamesList[predictedIndex])

        # ACTUAL ACCURACY, broadTYPE ACCURACY, AGE ACCURACY
        typeAndAgeCorrect = 0
        typeCorrect = 0
        broadTypeCorrect = 0
        broadTypeAndAgeCorrect = 0
        typeAndNearAgeCorrect = 0
        broadTypeAndNearAgeCorrect = 0
        for i in range(len(testTypeNames)):
            predictedIndex = np.argmax(yy[i])
            testBroadType = testTypeNames[i][0:2]
            actualBroadType = typeNamesList[predictedIndex][0:2]
            if testTypeNames[i][0:3] == 'IIb':
                testBroadType = 'Ib'
            if typeNamesList[predictedIndex][0:3] == 'IIb':
                actualBroadType = 'Ib'
            testType = testTypeNames[i].split(': ')[0]
            actualType = typeNamesList[predictedIndex].split(': ')[0]
            testAge = testTypeNames[i].split(': ')[1]
            actualAge = typeNamesList[predictedIndex].split(': ')[1]
            nearTestAge = testAge.split(' to ')

            if testTypeNames[i] == typeNamesList[predictedIndex]:
                typeAndAgeCorrect += 1
            if testType == actualType:  # correct type
                typeCorrect += 1
                if (nearTestAge[0] in actualAge) or (nearTestAge[1] in actualAge):  # check if the age is in the neigbouring bin
                    typeAndNearAgeCorrect += 1  # all correct except nearby bin
            if testBroadType == actualBroadType:  # correct broadtype
                broadTypeCorrect += 1
                if testAge == actualAge:
                    broadTypeAndAgeCorrect += 1
                if (nearTestAge[0] in actualAge) or (nearTestAge[1] in actualAge):  # check if the age is in the neigbouring bin
                    broadTypeAndNearAgeCorrect += 1  # Broadtype and nearby bin

        typeAndAgeAccuracy = float(typeAndAgeCorrect) / len(testTypeNames)
        typeAccuracy = float(typeCorrect) / len(testTypeNames)
        broadTypeAccuracy = float(broadTypeCorrect) / len(testTypeNames)
        broadTypeAndAgeAccuracy = float(broadTypeAndAgeCorrect) / len(testTypeNames)
        typeAndNearAgeAccuracy = float(typeAndNearAgeCorrect) / len(testTypeNames)
        broadTypeAndNearAgeAccuracy = float(broadTypeAndNearAgeCorrect) / len(testTypeNames)

        print("typeAndAgeAccuracy : " + str(typeAndAgeAccuracy))
        print("typeAccuracy : " + str(typeAccuracy))
        print("broadTypeAccuracy : " + str(broadTypeAccuracy))
        print("broadTypeAndAgeAccuracy: " + str(broadTypeAndAgeAccuracy))
        print("typeAndNearAgeAccuracy : " + str(typeAndNearAgeAccuracy))
        print("broadTypeAndNearAgeAccuracy : " + str(broadTypeAndNearAgeAccuracy))

        # SAVE THE MODEL
        saveFilename = "data_files/model_trainedAtZeroZ.ckpt"
        saver = tf.train.Saver()
        save_path = saver.save(sess, saveFilename)
        print("Model saved in file: %s" % save_path)

        modelFilenames = [saveFilename+'.index', saveFilename+'.meta', saveFilename+'.data-00000-of-00001']

        return modelFilenames

if __name__ == '__main__':
    t1 = time.time()
    savedFilenames = train_model()
    t2 = time.time()
    print("time spent: {0:.2f}".format(t2 - t1))

