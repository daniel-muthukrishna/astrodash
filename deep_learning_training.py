"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

#10 classes
#784 (28x28) pixels in each image
#None (55000) images

####################################################
#mnist = read_data_sets("MNIST_data/", one_hot=True)

import matplotlib.pyplot as plt
import numpy as np
import itertools
import tensorflow as tf
#import create_arrays

inputLoaded = np.load('input_data.npz')
inputImages = inputLoaded['inputImages']
inputLabels = inputLoaded['inputLabels']
inputFilenames = inputLoaded['inputFilenames']
inputTypeNames = inputLoaded['inputTypeNames']
inputRedshifts = inputLoaded['inputRedshifts']

snidtempfilelist = r'/home/dan/Desktop/SNClassifying/templates/templist'
loaded = np.load('file_w_ages2.npz')
trainImages = loaded['trainImages']
trainLabels = loaded['trainLabels']
trainFilenames = loaded['trainFilenames']
trainTypeNames = loaded['trainTypeNames']
testImages = loaded['testImages']
testLabels = loaded['testLabels']
testFilenames = loaded['testFilenames']
testTypeNames = loaded['testTypeNames']
typeNamesList = loaded['typeNamesList']
#validateImages = sortData[2][0]
#validateLabels = sortData[2][1]

testLabels1 = []
trainLabels1 = []
inputLabels1 = []


print("Completed creatingArrays")

N = 1024
ntypes = len(testLabels[0])
print(ntypes)

a = []

#IMPLEMENTING THE REGRESSSION
x = tf.placeholder(tf.float32, [None, N])

W = tf.Variable(tf.zeros([N, ntypes]))
b = tf.Variable(tf.zeros([ntypes]))

y = tf.nn.softmax(tf.matmul(x, W) + b)


#TRAINING
y_ = tf.placeholder(tf.float32, [None, ntypes]) #correct answers

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

batch_xs1 = trainImages
batch_ys1 = trainLabels
print(sess.run(y, feed_dict={x: batch_xs1, y_: batch_ys1}))

#Train 1000 times
trainImagesCycle = itertools.cycle(trainImages)
trainLabelsCycle = itertools.cycle(trainLabels)
for i in range(600):
    batch_xs = np.array(list(itertools.islice(trainImagesCycle, 5000*i, 5000*i+5000)))
    batch_ys = np.array(list(itertools.islice(trainLabelsCycle, 5000*i, 5000*i+5000)))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if (i % 100 == 1):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        testacc = sess.run(accuracy, feed_dict={x: testImages, y_: testLabels})
        trainacc = sess.run(accuracy, feed_dict={x: trainImages, y_: trainLabels})
        a.append(testacc)
        print(str(testacc) + " " + str(trainacc))

batch_xs1 = testImages
batch_ys1 = testLabels
print(sess.run(y, feed_dict={x: batch_xs1, y_: batch_ys1}))
yy = sess.run(y, feed_dict={x: testImages, y_: testLabels})

#EVALUATING THE MODEL
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: testImages, y_: testLabels}))


yy = sess.run(y, feed_dict={x: testImages, y_: testLabels})
cp = sess.run(correct_prediction, feed_dict={x: testImages, y_: testLabels})
print(cp)
for i in range(len(cp)):
    if (cp[i] == False):
        predictedIndex = np.argmax(yy[i])
        print(i, testTypeNames[i], typeNamesList[predictedIndex])

#ACTUAL ACCURACY, SUBTYPE ACCURACY, AGE ACCURACY


############################################################
yInputRedshift = sess.run(y, feed_dict={x: inputImages})
print(yInputRedshift)
print(sess.run(accuracy, feed_dict={x: inputImages, y_: inputLabels}))

#Create List of Best Types
bestForEachType = np.zeros((ntypes,3))
index = np.zeros(ntypes)
for i in range(len(yInputRedshift)):
    prob = yInputRedshift[i]
    z = inputRedshifts[i]
    bestIndex = np.argmax(prob)
    if prob[bestIndex] > bestForEachType[bestIndex][2]:
        bestForEachType[bestIndex][2] = prob[bestIndex]
        bestForEachType[bestIndex][1] = z
        bestForEachType[bestIndex][0] = bestIndex #inputTypeNames
        index[bestIndex] = i

idx = np.argsort(bestForEachType[:,2])
bestForEachType = bestForEachType[idx[::-1]]

print ("Type          Redshift      Rel. Prob.")
print(bestForEachType)
for i in range(10):#ntypes):
    bestIndex = bestForEachType[i][0]
    print(typeNamesList[bestIndex] + '\t' + bestForEachType[i][1:])



#Plot Each Best Type at corresponding best redshift
for c in range(2):#ntypes):
    for i in range(0,len(trainImages)):
        if (trainLabels[i][c] == 1):
            print(i)
            plt.plot(trainImages[i])
            plt.plot(inputImages[index[c]])
            plt.title(str(bestForEachType[c][0])+": " + str(bestForEachType[c][1]))
            plt.show()
            break
        

#Plot Probability vs redshift for each class
redshiftGraphs = [[[],[]] for i in range(ntypes)]
for c in range(2):#ntypes):
    redshiftGraphs[c][0] = inputRedshifts
    redshiftGraphs[c][1] = yInputRedshift[:,c]
    plt.plot(redshiftGraphs[c][0],redshiftGraphs[c][1])
    plt.xlabel("z")
    plt.ylabel("Probability")
    bestIndex = bestForEachType[c][0]
    plt.title("Type: " + typeNamesList[bestIndex])
    plt.show()

redshiftGraphs = np.array(redshiftGraphs)



'''
for i in range(len(testImages)):
	if (testLabels[i][13] == 1):
		print i
		plt.plot(testImages[i])
'''
#[ 115.   13.   14.   13.    8.   26.   26.    9.    4.   10.   18.    3.  8.   15.]
