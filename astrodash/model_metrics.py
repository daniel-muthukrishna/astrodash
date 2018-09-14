import os
import pickle
import matplotlib.pyplot as plt
import itertools
import numpy as np
import tensorflow as tf
from astrodash.multilayer_convnet import convnet_variables


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.RdBu, fig_dir='.', name='', fontsize_labels=15, fontsize_matrix=18):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    np.savetxt(os.path.join(fig_dir, 'confusion_matrix_raw_%s.csv' % name), cm)

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
    cb = plt.colorbar()
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=16)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=fontsize_labels)
    plt.yticks(tick_marks, classes, fontsize=fontsize_labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(abs(cm[i, j]), fmt), horizontalalignment="center",
                 color="white" if abs(cm[i, j]) > thresh else "black", fontsize=fontsize_matrix)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'confusion_matrix_%s.pdf' % name))


def get_aggregated_conf_matrix(aggregateIndexes, testLabels, predictedLabels):
    testLabelsAggregated = np.digitize(testLabels, aggregateIndexes) - 1
    predictedLabelsAggregated = np.digitize(predictedLabels, aggregateIndexes) - 1
    confMatrixAggregated = tf.confusion_matrix(testLabelsAggregated, predictedLabelsAggregated).eval()
    np.set_printoptions(precision=2)
    print(confMatrixAggregated)

    return confMatrixAggregated


def calc_model_metrics(modelFilename, testLabels, testImages, testTypeNames, typeNamesList, snTypes=None, fig_dir='.'):
    tf.reset_default_graph()
    nw = len(testImages[0])
    nBins = len(typeNamesList)
    imWidthReduc = 8
    imWidth = 32  # Image size and width

    x, y_, keep_prob, y_conv, W, b = convnet_variables(imWidth, imWidthReduc, nw, nBins)

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

        # Aggregate age conf matrix
        aggregateAgesIndexes = np.arange(0, nBins + 1, int(nBins / len(snTypes)))
        confMatrixAggregateAges = get_aggregated_conf_matrix(aggregateAgesIndexes, testLabels, predictedLabels)
        classnames = np.copy(snTypes)
        if confMatrixAggregateAges.shape[0] < len(classnames):
            classnames = classnames[:-1]
        plot_confusion_matrix(confMatrixAggregateAges, classes=classnames, normalize=True, title='', fig_dir=fig_dir, name='aggregate_ages', fontsize_labels=15, fontsize_matrix=16)

        # Aggregate age and subtypes conf matrix
        aggregateSubtypesIndexes = np.array([0, 108, 180, 234, 306])
        broadTypes = ['Ia', 'Ib', 'Ic', 'II']
        confMatrixAggregateSubtypes = get_aggregated_conf_matrix(aggregateSubtypesIndexes, testLabels, predictedLabels)
        plot_confusion_matrix(confMatrixAggregateSubtypes, classes=broadTypes, normalize=True, title='', fig_dir=fig_dir, name='aggregate_subtypes', fontsize_labels=30, fontsize_matrix=30)
        # plt.show()

    np.set_printoptions(precision=2)
    print(confMatrix)
    plot_confusion_matrix(confMatrix, classes=typeNamesList, normalize=True, title='', fig_dir=fig_dir, name='all', fontsize_labels=2, fontsize_matrix=1)

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


def main():
    dirModel = "/Users/danmuth/PycharmProjects/astrodash/data_files_train80_splitspectra_zeroZ/"
    modelFilename = dirModel + "tensorflow_model.ckpt"

    fig_dir = os.path.join('..', 'Figures', 'zeroZ_train80_splitspectra')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    with open(os.path.join(dirModel, "training_params.pickle"), 'rb') as f:
        pars = pickle.load(f)
    snTypes = pars['typeList']

    dirTestSet = "/Users/danmuth/PycharmProjects/astrodash/data_files_train80_splitspectra_zeroZ/training_set/"
    testImagesAll = np.load(dirTestSet + 'testImages.npy', mmap_mode='r')
    testLabelsAll = np.load(dirTestSet + 'testLabels.npy', mmap_mode='r')
    typeNamesList = np.load(dirTestSet + 'typeNamesList.npy')
    testTypeNamesAll = np.load(dirTestSet + 'testTypeNames.npy')

    calc_model_metrics(modelFilename, testLabelsAll[:50000], testImagesAll[:50000], testTypeNamesAll[:50000], typeNamesList, snTypes, fig_dir=fig_dir)


if __name__ == '__main__':
    main()
