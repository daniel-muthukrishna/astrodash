import numpy as np
import tensorflow as tf
from dash.multilayer_convnet import convnet_variables


def calc_model_statistics(modelFilename, testImages, testTypeNames, typeNamesList):
    nw = len(testImages[0])
    nBins = len(typeNamesList)
    imWidthReduc = 8
    imWidth = 32  # Image size and width

    x, y_, keep_prob, y_conv = convnet_variables(imWidth, imWidthReduc, nw, nBins)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, modelFilename)

        yy = y_conv.eval(feed_dict={x: testImages, keep_prob: 1.0})

    # ACTUAL ACCURACY, broadTYPE ACCURACY, AGE ACCURACY
    typeAndAgeCorrect = 0
    typeCorrect = 0
    broadTypeCorrect = 0
    broadTypeAndAgeCorrect = 0
    typeAndNearAgeCorrect = 0
    broadTypeAndNearAgeCorrect = 0
    for i in range(len(testTypeNames)):
        predictedIndex = np.argmax(yy[i])

        classification = testTypeNames[i].split(': ')
        if len(classification) == 2:
            testType, testAge = classification
        else:
            testGalType, testType, testAge = classification
        actual = typeNamesList[predictedIndex].split(': ')
        if len(actual) == 2:
            actualType, actualAge = actual
        else:
            actualGalType, actualType, actualAge = actual

        testBroadType = testType[0:2]
        actualBroadType = actualType[0:2]
        if testType[0:3] == 'IIb':
            testBroadType = 'Ib'
        if actualType[0:3] == 'IIb':
            actualBroadType = 'Ib'
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


if __name__ == '__main__':
    dirModel = "/Users/danmuth/PycharmProjects/DASH/dash/models_other/zeroZ/data_files_zeroZ_withHost_withNoise/"
    modelFilename = dirModel + "tensorflow_model.ckpt"

    dirTestSet = "/Users/danmuth/PycharmProjects/DASH/dash/models_other/zeroZ/data_files_zeroZ_withHost_withNoise/training_set/"
    testImages = np.load(dirTestSet + 'testImages.npy', mmap_mode='r')
    # testLabelsIndexes = np.load(dir1 + 'testLabels.npy', mmap_mode='r')
    typeNamesList = np.load(dirTestSet + 'typeNamesList.npy')
    testTypeNames = np.load(dirTestSet + 'testTypeNames.npy')

    calc_model_statistics(modelFilename, testImages[:1000], testTypeNames[:1000], typeNamesList)
