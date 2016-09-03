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


loaded = np.load('type_age_atRedshiftZero.npz')
trainImages = loaded['trainImages']
trainLabels = loaded['trainLabels']
#trainFilenames = loaded['trainFilenames']
#trainTypeNames = loaded['trainTypeNames']
testImages = loaded['testImages']
testLabels = loaded['testLabels']
#testFilenames = loaded['testFilenames']
testTypeNames = loaded['testTypeNames']
typeNamesList = loaded['typeNamesList']
#validateImages = sortData[2][0]
#validateLabels = sortData[2][1]


print("Completed creatingArrays")

N = 1024
ntypes = len(testLabels[0])
imWidth = 32 #Image size and width
imWidthReduc = 8

a = []

sess = tf.InteractiveSession()

#WEIGHT INITIALISATION
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#CONVOLUTION AND POOLING
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, N])
y_ = tf.placeholder(tf.float32, shape=[None, ntypes])

#FIRST CONVOLUTIONAL LAYER
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,imWidth,imWidth,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#SECOND CONVOLUTIONAL LAYER
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#DENSELY CONNECTED LAYER
W_fc1 = weight_variable([imWidthReduc * imWidthReduc * 64, N])
b_fc1 = bias_variable([N])
h_pool2_flat = tf.reshape(h_pool2, [-1, imWidthReduc*imWidthReduc*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#DROPOUT
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#READOUT LAYER
W_fc2 = weight_variable([N, ntypes])
b_fc2 = bias_variable([ntypes])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

########
##batch_xs1 = trainImages
##batch_ys1 = trainLabels
##print(sess.run(y_conv, feed_dict={x: batch_xs1, y_: batch_ys1}))
########

#TRAIN AND EVALUATE MODEL
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

trainImagesCycle = itertools.cycle(trainImages)
trainLabelsCycle = itertools.cycle(trainLabels)
for i in range(100000):
    batch_xs = np.array(list(itertools.islice(trainImagesCycle, 50*i, 50*i+50)))
    batch_ys = np.array(list(itertools.islice(trainLabelsCycle, 50*i, 50*i+50)))
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    if (i % 100 == 0):
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        testacc = accuracy.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0})
        print("test accuracy %g"%(testacc))
        a.append(testacc)
print("test accuracy %g"%accuracy.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0}))

yy = y_conv.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0})
cp = correct_prediction.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0})
print(cp)
for i in range(len(cp)):
    if (cp[i] == False):
        predictedIndex = np.argmax(yy[i])
        print(i, testTypeNames[i], typeNamesList[predictedIndex])


#ACTUAL ACCURACY, SUBTYPE ACCURACY, AGE ACCURACY
typeAndAgeCorrect = 0
typeCorrect = 0
subTypeCorrect = 0
subTypeAndAgeCorrect = 0
typeAndNearAgeCorrect = 0
subTypeAndNearAgeCorrect = 0
for i in range(len(testTypeNames)):
    predictedIndex = np.argmax(yy[i])
    testSubType = testTypeNames[i][0:2]
    actualSubType = typeNamesList[predictedIndex][0:2]
    if testTypeNames[i][0:3] == 'IIb':
        testSubType = 'Ib'
    if typeNamesList[predictedIndex][0:3] == 'IIb':
        actualSubType = 'Ib'
    testType = testTypeNames[i].split(': ')[0]
    actualType = typeNamesList[predictedIndex].split(': ')[0]
    testAge = testTypeNames[i].split(': ')[1]
    actualAge = typeNamesList[predictedIndex].split(': ')[1]
    nearTestAge = testAge.split(' to ')
    
    if (testTypeNames[i] == typeNamesList[predictedIndex]):
        typeAndAgeCorrect += 1
    if (testType == actualType): #correct type
        typeCorrect += 1
        if ((nearTestAge[0] in actualAge) or (nearTestAge[1] in actualAge)): #check if the age is in the neigbouring bin
            typeAndNearAgeCorrect += 1 #all correct except nearby bin
    if (testSubType == actualSubType): #correct subtype
        subTypeCorrect += 1
        if testAge == actualAge:
            subTypeAndAgeCorrect += 1
        if ((nearTestAge[0] in actualAge) or (nearTestAge[1] in actualAge)): #check if the age is in the neigbouring bin
            subTypeAndNearAgeCorrect += 1 #subtype and nearby bin

typeAndAgeAccuracy = float(typeAndAgeCorrect)/len(testTypeNames)
typeAccuracy = float(typeCorrect)/len(testTypeNames)
subTypeAccuracy = float(subTypeCorrect)/len(testTypeNames)
subTypeAndAgeAccuracy = float(subTypeAndAgeCorrect)/len(testTypeNames)
typeAndNearAgeAccuracy = float(typeAndNearAgeCorrect)/len(testTypeNames)
subTypeAndNearAgeAccuracy = float(subTypeAndNearAgeCorrect)/len(testTypeNames)

print("typeAndAgeAccuracy : " + str(typeAndAgeAccuracy))
print("typeAccuracy : " + str(typeAccuracy))
print("subTypeAccuracy : " + str(subTypeAccuracy))
print("subTypeAndAgeAccuracy: " + str(subTypeAndAgeAccuracy))
print("typeAndNearAgeAccuracy : " + str(typeAndNearAgeAccuracy))
print("subTypeAndNearAgeAccuracy : " + str(subTypeAndNearAgeAccuracy))
    

#SAVE THE MODEL
saver = tf.train.Saver()
save_path = saver.save(sess, "model_trainedAtZeroZ.ckpt")
print("Model saved in file: %s" % save_path)
