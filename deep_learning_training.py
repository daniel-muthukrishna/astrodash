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
nBins = len(testLabels[0])
print(nBins)

a = []

#IMPLEMENTING THE REGRESSSION
x = tf.placeholder(tf.float32, [None, N])

W = tf.Variable(tf.zeros([N, nBins]))
b = tf.Variable(tf.zeros([nBins]))

y = tf.nn.softmax(tf.matmul(x, W) + b)


#TRAINING
y_ = tf.placeholder(tf.float32, [None, nBins]) #correct answers

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#batch_xs1 = trainImages
#batch_ys1 = trainLabels
#print(sess.run(y, feed_dict={x: batch_xs1, y_: batch_ys1}))

#Train 1000 times
trainImagesCycle = itertools.cycle(trainImages)
trainLabelsCycle = itertools.cycle(trainLabels)
for i in range(4000):
    batch_xs = np.array(list(itertools.islice(trainImagesCycle, 5000*i, 5000*i+5000)))
    batch_ys = np.array(list(itertools.islice(trainLabelsCycle, 5000*i, 5000*i+5000)))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if (i % 100 == 1):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        testacc = sess.run(accuracy, feed_dict={x: testImages, y_: testLabels})
        trainacc = sess.run(accuracy, feed_dict={x: trainImages[0:1000], y_: trainLabels[0:1000]})
        a.append(testacc)
        print(i, str(testacc) + " " + str(trainacc))

batch_xs1 = testImages
batch_ys1 = testLabels
#print(sess.run(y, feed_dict={x: batch_xs1, y_: batch_ys1}))
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
save_path = saver.save(sess, "model.ckpt")
print("Model saved in file: %s" % save_path)


