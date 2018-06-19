import os
import pickle
import matplotlib.pyplot as plt
import itertools
import numpy as np
import tensorflow as tf
from dash.multilayer_convnet import convnet_variables


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.RdBu, fig_dir='.', name=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    # Multiply off diagonal by -1
    off_diag = ~np.eye(cm.shape[0], dtype=bool)
    cm[off_diag] *= -1
    np.savetxt(os.path.join(fig_dir, 'confusion_matrix_%s.csv' % name), cm)
    print(cm)

    fig = plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=-1, vmax=1)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(abs(cm[i, j]), fmt), horizontalalignment="center",
                 color="white" if abs(cm[i, j]) > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'confusion_matrix_%s' % name))


def calc_model_statistics(modelFilename, testLabels, testImages, testTypeNames, typeNamesList, typeNamesBroad=None):
    tf.reset_default_graph()
    nw = len(testImages[0])
    nBins = len(typeNamesList)
    imWidthReduc = 8
    imWidth = 32  # Image size and width

    x, y_, keep_prob, y_conv = convnet_variables(imWidth, imWidthReduc, nw, nBins)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, modelFilename)

        yy = y_conv.eval(feed_dict={x: testImages, keep_prob: 1.0})

        # CONFUSION MATRIX
        predictedLabels = []
        for i, name in enumerate(testTypeNames):
            predictedLabels.append(np.argmax(yy[i]))
        predictedLabels = np.array(predictedLabels)
        confMatrix = tf.confusion_matrix(testLabels, predictedLabels).eval()

        if typeNamesBroad is not None:
            broadTypeLabels = np.arange(0, nBins+1, int(nBins/len(typeNamesBroad)))
            testLabelsBroad = np.digitize(testLabels, broadTypeLabels) - 1
            predictedLabelsBroad = np.digitize(predictedLabels, broadTypeLabels) - 1
            confMatrixBroad = tf.confusion_matrix(testLabelsBroad, predictedLabelsBroad).eval()
            np.set_printoptions(precision=2)
            print(confMatrixBroad)
            plot_confusion_matrix(confMatrixBroad, classes=typeNamesBroad, normalize=True, title='Normalized confusion matrix', fig_dir='.', name='broad')
            # plt.show()

    # np.set_printoptions(precision=2)
    # print(confMatrix)
    # plot_confusion_matrix(confMatrix, classes=typeNamesList, normalize=True, title='Normalized confusion matrix', fig_dir='.', name='all')

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

    with open(os.path.join(dirModel, "training_params.pickle"), 'rb') as f:
        pars = pickle.load(f)
    typeNamesBroad = pars['typeList']

    dirTestSet = "/Users/danmuth/PycharmProjects/DASH/dash/models_other/zeroZ/data_files_zeroZ_withHost_withNoise/training_set/"
    testImages = np.load(dirTestSet + 'testImages.npy', mmap_mode='r')
    testLabels = np.load(dirTestSet + 'testLabels.npy', mmap_mode='r')
    typeNamesList = np.load(dirTestSet + 'typeNamesList.npy')
    testTypeNames = np.load(dirTestSet + 'testTypeNames.npy')

    calc_model_statistics(modelFilename, testLabels[:1000], testImages[:1000], testTypeNames[:1000], typeNamesList, typeNamesBroad)
