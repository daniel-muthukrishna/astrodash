import numpy as np
import itertools
import tensorflow as tf
from multilayer_convnet import ConvNetLayer

loaded = np.load('type_age_atRedshiftZero.npz')
trainImages = loaded['trainImages']
trainLabels = loaded['trainLabels']
# trainFilenames = loaded['trainFilenames']
# trainTypeNames = loaded['trainTypeNames']
testImages = loaded['testImages']
testLabels = loaded['testLabels']
# testFilenames = loaded['testFilenames']
testTypeNames = loaded['testTypeNames']
typeNamesList = loaded['typeNamesList']
# validateImages = sortData[2][0]
# validateLabels = sortData[2][1]


print("Completed creatingArrays")

N = 1024
ntypes = len(testLabels[0])
imWidth = 32  # Image size and width
imWidthReduc = 8

a = []

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, N])
y_ = tf.placeholder(tf.float32, shape=[None, ntypes])

x_image = tf.reshape(x, [-1, imWidth, imWidth, 1])

Layer1 = ConvNetLayer(N, ntypes, imWidth, imWidthReduc)
h_pool1 = Layer1.build_layer(x_image, 1, 32)

Layer2 = ConvNetLayer(N, ntypes, imWidth, imWidthReduc)
h_pool2 = Layer2.build_layer(h_pool1, 32, 64)
h_fc1 = Layer2.connect_layers(h_pool2, 64, 1)

Layer3 = ConvNetLayer(N, ntypes, imWidth, imWidthReduc)
h_pool3 = Layer3.build_layer(h_pool2, 64, 64)
h_fc2 = Layer3.connect_layers(h_pool3, 64, 2)


# READOUT LAYER
keep_prob, h_fc2_drop = Layer3.dropout(h_fc2)
W_fc3, b_fc3 = Layer3.readout_layer()

y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)


# TRAIN AND EVALUATE MODEL
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

trainImagesCycle = itertools.cycle(trainImages)
trainLabelsCycle = itertools.cycle(trainLabels)
for i in range(30000):
    batch_xs = np.array(list(itertools.islice(trainImagesCycle, 50 * i, 50 * i + 50)))
    batch_ys = np.array(list(itertools.islice(trainLabelsCycle, 50 * i, 50 * i + 50)))
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    if (i % 100 == 0):
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        testacc = accuracy.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0})
        print("test accuracy %g" % (testacc))
        a.append(testacc)
print("test accuracy %g" % accuracy.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0}))

yy = y_conv.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0})
cp = correct_prediction.eval(feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0})
print(cp)
for i in range(len(cp)):
    if (cp[i] == False):
        predictedIndex = np.argmax(yy[i])
        print(i, testTypeNames[i], typeNamesList[predictedIndex])

# ACTUAL ACCURACY, SUBTYPE ACCURACY, AGE ACCURACY
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
    if (testType == actualType):  # correct type
        typeCorrect += 1
        if ((nearTestAge[0] in actualAge) or (
            nearTestAge[1] in actualAge)):  # check if the age is in the neigbouring bin
            typeAndNearAgeCorrect += 1  # all correct except nearby bin
    if (testSubType == actualSubType):  # correct subtype
        subTypeCorrect += 1
        if testAge == actualAge:
            subTypeAndAgeCorrect += 1
        if ((nearTestAge[0] in actualAge) or (
            nearTestAge[1] in actualAge)):  # check if the age is in the neigbouring bin
            subTypeAndNearAgeCorrect += 1  # subtype and nearby bin

typeAndAgeAccuracy = float(typeAndAgeCorrect) / len(testTypeNames)
typeAccuracy = float(typeCorrect) / len(testTypeNames)
subTypeAccuracy = float(subTypeCorrect) / len(testTypeNames)
subTypeAndAgeAccuracy = float(subTypeAndAgeCorrect) / len(testTypeNames)
typeAndNearAgeAccuracy = float(typeAndNearAgeCorrect) / len(testTypeNames)
subTypeAndNearAgeAccuracy = float(subTypeAndNearAgeCorrect) / len(testTypeNames)

print("typeAndAgeAccuracy : " + str(typeAndAgeAccuracy))
print("typeAccuracy : " + str(typeAccuracy))
print("subTypeAccuracy : " + str(subTypeAccuracy))
print("subTypeAndAgeAccuracy: " + str(subTypeAndAgeAccuracy))
print("typeAndNearAgeAccuracy : " + str(typeAndNearAgeAccuracy))
print("subTypeAndNearAgeAccuracy : " + str(subTypeAndNearAgeAccuracy))

# SAVE THE MODEL
saver = tf.train.Saver()
save_path = saver.save(sess, "model_trainedAtZeroZ_multilayer.ckpt")
print("Model saved in file: %s" % save_path)
