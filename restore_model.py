import tensorflow as tf
import numpy as np
import pickle
from input_spectra import *

loaded = np.load('/home/dan/Desktop/SNClassifying_Pre-alpha/type_age_atRedshiftZero.npz')
trainImages = loaded['trainImages']
trainLabels = loaded['trainLabels']
##trainFilenames = loaded['trainFilenames']
##trainTypeNames = loaded['trainTypeNames']
##testImages = loaded['testImages']
##testLabels = loaded['testLabels']
##testFilenames = loaded['testFilenames']
##testTypeNames = loaded['testTypeNames']
##typeNamesList = loaded['typeNamesList']

##inputLoaded = np.load('/home/dan/Desktop/SNClassifying_Pre-alpha/input_data.npz')
##inputImages = inputLoaded['inputImages']
##inputLabels = inputLoaded['inputLabels']
##inputFilenames = inputLoaded['inputFilenames']
##inputTypeNames = inputLoaded['inputTypeNames']
##inputRedshifts = inputLoaded['inputRedshifts']
##typeNamesList = inputLoaded['typeNamesList']


class LoadInputSpectra(object):
    def __init__(self, inputFilename, minZ, maxZ):
        with open('/home/dan/Desktop/SNClassifying_Pre-alpha/training_params.pickle') as f:
            nTypes, w0, w1, nw, minAge, maxAge, ageBinSize, typeList = pickle.load(f)

        self.nw = nw
        self.nTypes = nTypes
        
        self.inputSpectra = InputSpectra(inputFilename, minZ, maxZ, nTypes, minAge, maxAge, ageBinSize, w0, w1, nw, typeList)

        self.inputImages, self.inputFilenames, self.inputRedshifts, self.typeNamesList = self.inputSpectra.redshifting()
        self.nBins = len(self.typeNamesList)

    def input_spectra(self):
        return self.inputImages, self.inputRedshifts, self.typeNamesList, int(self.nw), self.nBins
        
class RestoreModel(object):
    def __init__(self, modelFilename, inputImages, nw, nBins):
        self.reset()
        
        self.modelFilename = modelFilename
        self.inputImages = inputImages
##        self.x = tf.placeholder(tf.float32, [None, nw])
##        self.W = tf.Variable(tf.zeros([nw, nBins]))
##        self.b = tf.Variable(tf.zeros([nBins]))
##        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
##        self.y_ = tf.placeholder(tf.float32, [None, nBins])

        self.nw = nw
        self.nBins = nBins
        self.imWidthReduc = 8
        self.imWidth = 32 #Image size and width
        
        self.x = tf.placeholder(tf.float32, shape=[None, nw])
        self.y_ = tf.placeholder(tf.float32, shape=[None, nBins])

        #FIRST CONVOLUTIONAL LAYER
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        x_image = tf.reshape(self.x, [-1,self.imWidth,self.imWidth,1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

         #SECOND CONVOLUTIONAL LAYER
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)


        #DENSELY CONNECTED LAYER
        W_fc1 = self.weight_variable([self.imWidthReduc * self.imWidthReduc * 64, 1024])
        b_fc1 = self.bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, self.imWidthReduc*self.imWidthReduc*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        #DROPOUT
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        #READOUT LAYER
        W_fc2 = self.weight_variable([1024, nBins])
        b_fc2 = self.bias_variable([nBins])

        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


        self.saver = tf.train.Saver()

    #WEIGHT INITIALISATION
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    #CONVOLUTION AND POOLING
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


        self.saver = tf.train.Saver()

    def restore_variables(self):
        with tf.Session() as sess:
            self.saver.restore(sess, self.modelFilename)
##            softmax = sess.run(self.y, feed_dict={self.x: self.inputImages})
##            correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
##            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
##            print(softmax)
            softmax = self.y_conv.eval(feed_dict={self.x: self.inputImages, self.keep_prob: 1.0})
            print(softmax)
            
        return softmax

    def reset(self):
        tf.reset_default_graph()


class RestoreModelSingleRedshift(object):
    def __init__(self, modelFilename, inputImage, nw, nBins):
        self.reset()
        
        self.modelFilename = modelFilename
        self.inputImage = inputImage
        self.nw = nw
        self.nBins = nBins
        self.imWidthReduc = 8
        self.imWidth = 32 #Image size and width
        
        self.x = tf.placeholder(tf.float32, shape=[None, nw])
        self.y_ = tf.placeholder(tf.float32, shape=[None, nBins])

        #FIRST CONVOLUTIONAL LAYER
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        x_image = tf.reshape(self.x, [-1,self.imWidth,self.imWidth,1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

         #SECOND CONVOLUTIONAL LAYER
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)


        #DENSELY CONNECTED LAYER
        W_fc1 = self.weight_variable([self.imWidthReduc * self.imWidthReduc * 64, 1024])
        b_fc1 = self.bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, self.imWidthReduc*self.imWidthReduc*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        #DROPOUT
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        #READOUT LAYER
        W_fc2 = self.weight_variable([1024, nBins])
        b_fc2 = self.bias_variable([nBins])

        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


        self.saver = tf.train.Saver()

    #WEIGHT INITIALISATION
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    #CONVOLUTION AND POOLING
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def restore_variables(self):
        with tf.Session() as sess:
            self.saver.restore(sess, self.modelFilename)

            softmax = self.y_conv.eval(feed_dict={self.x: self.inputImage, self.keep_prob: 1.0})
            print(softmax)
            
        return softmax

    def reset(self):
        tf.reset_default_graph()
        

        

class BestTypesListSingleRedshift(object):
    def __init__(self, modelFilename, inputImage, typeNamesList, nw, nBins):
        self.modelFilename = modelFilename
        self.inputImage = inputImage
        self.typeNamesList = typeNamesList
        self.nBins = nBins

        self.restoreModel = RestoreModelSingleRedshift(self.modelFilename, self.inputImage, nw, nBins)
        self.typeNamesList = np.array(typeNamesList)
        self.inputImage = self.inputImage[0]
        self.softmax = self.restoreModel.restore_variables()[0]
        print(len(self.softmax))

        self.bestTypes, self.idx, self.softmaxOrdered = self.create_list()


    def create_list(self):
        idx = np.argsort(self.softmax) #list of the index of the highest probabiites
        bestTypes = self.typeNamesList[idx[::-1]] #reordered in terms of softmax probability columns
        print idx
        print bestTypes
        print self.softmax[idx[::-1]]
        print self.softmax
        return bestTypes, idx, self.softmax[idx[::-1]]

    def plot_best_types(self):
        inputFluxes = []
        templateFluxes = []
        for j in range(20):#len(self.bestTypes)): #index of best Types in order
            c = self.idx[::-1][j]
            print c
            typeName = self.typeNamesList[c] #Name of best type
            for i in range(len(trainLabels)): #Checking through templates
                if (trainLabels[i][c] == 1):    #to find template for the best Type
                    templateFlux = trainImages[i]  #plot template
                    inputFlux = self.inputImage #Plot inputImage at red
                    print c
                    break
            if (i == len(trainLabels)-1):
                print("No Template") #NEED TO GET TEMPLATE PLOTS IN A BETTER WAY
                templateFlux = np.zeros(len(trainImages[0]))
                inputFlux = self.inputImage

            templateFluxes.append(templateFlux)
            inputFluxes.append(inputFlux)
        print self.idx[::-1]
            
        templateFluxes = np.array(templateFluxes)
        inputFluxes = np.array(inputFluxes)
        return templateFluxes, inputFluxes

    

class BestTypesList(object):
    def __init__(self, modelFilename, inputImages, inputRedshifts, typeNamesList, nw, nBins):
        self.modelFilename = modelFilename
        self.inputImages = inputImages
        self.inputRedshifts = inputRedshifts
        self.typeNamesList = typeNamesList
        self.nBins = nBins
        
        self.restoreModel = RestoreModel(self.modelFilename, self.inputImages, nw, nBins)
        self.yInputRedshift = self.restoreModel.restore_variables() #softmax at each redshift
        print(len(self.yInputRedshift))

        self.bestForEachType, self.redshiftIndex = self.create_list()

    def create_list(self):
        bestForEachType = np.zeros((self.nBins,3))
        redshiftIndex = np.zeros(self.nBins)        #best redshift index for each type
        for i in range(len(self.inputRedshifts)): #for each redshift
            z = self.inputRedshifts[i]       #redshift
            softmax = self.yInputRedshift[i] #softmax probabilities at particular redshift
            bestIndex = np.argmax(softmax) #index of the Best type at redshift
            if softmax[bestIndex] > bestForEachType[bestIndex][2]: #if relProb of best type at this redshift is better than the relProb of this type at any redshift
                bestForEachType[bestIndex][2] = softmax[bestIndex]
                bestForEachType[bestIndex][1] = z
                bestForEachType[bestIndex][0] = bestIndex
                redshiftIndex[bestIndex] = i
        idx = np.argsort(bestForEachType[:,2]) #list of the index of the highest probabiites
        bestForEachType = bestForEachType[idx[::-1]] #reordered in terms of relProb columns
        
        return bestForEachType, redshiftIndex

    def print_list(self):
        return self.bestForEachType, self.redshiftIndex

    def plot_best_types(self):
        #bestForEachType, typeNamesList, redshiftIndex = self.create_list()
        inputFluxes = []
        templateFluxes = []
        for j in range(len(self.bestForEachType)): #index of best Types in order
            c = int(self.bestForEachType[:,0][j])
            typeName = self.typeNamesList[c] #Name of best type
            for i in range(0,len(trainLabels)): #Checking through templates
                if (trainLabels[i][c] == 1):    #to find template for the best Type
                    templateFlux = trainImages[i]  #plot template
                    inputFlux = self.inputImages[int(self.redshiftIndex[c])] #Pliot inputImage at red
                    print c, self.redshiftIndex[c]
                    #plt.title(typeName+ ": " + str(bestForEachType[c][1]))
                    break
            if (i == len(trainLabels)-1):
                #print("No Template") #NEED TO GET TEMPLATE PLOTS IN A BETTER WAY
                templateFlux = np.zeros(len(trainImages[0]))
                inputFlux = self.inputImages[int(self.redshiftIndex[c])]

            templateFluxes.append(templateFlux)
            inputFluxes.append(inputFlux)
            
        templateFluxes = np.array(templateFluxes)
        inputFluxes = np.array(inputFluxes)
        return templateFluxes, inputFluxes

    def redshift_graph(self):
        #bestForEachType, typeNamesList, redshiftIndex = self.create_list()
        self.redshiftGraphs = []#redshiftGraphs = [[[],[]] for i in range(ntypes)]
        for j in range(len(self.bestForEachType)): #[0:2] takes top 2 entries
            c = int(self.bestForEachType[:,0][j])
            typeName = self.typeNamesList[c]
            #redshiftGraphs[c][0] = self.inputRedshifts
            self.redshiftGraphs.append(self.yInputRedshift[:,c])
            

        self.redshiftGraphs = np.array(self.redshiftGraphs)

        return self.inputRedshifts, self.redshiftGraphs

#bestTypesList = BestTypesList("/tmp/model.ckpt")
#templateFluxes, inputFluxes = bestTypesList.plot_best_types()

##restoreModel = RestoreModel("/tmp/model.ckpt")
##yInputRedshift = restoreModel.restore_variables()
##
##
###Create List of Best Types
##bestForEachType = np.zeros((ntypes,3))
##redshiftIndex = np.zeros(ntypes) #best redshift index for each type
##for i in range(len(yInputRedshift)):
##    prob = yInputRedshift[i]
##    z = inputRedshifts[i]
##    bestIndex = np.argmax(prob)
##    if prob[bestIndex] > bestForEachType[bestIndex][2]:
##        bestForEachType[bestIndex][2] = prob[bestIndex]
##        bestForEachType[bestIndex][1] = z
##        bestForEachType[bestIndex][0] = bestIndex #inputTypeNames
##        redshiftIndex[bestIndex] = i
##
##idx = np.argsort(bestForEachType[:,2])
##bestForEachType = bestForEachType[idx[::-1]]
##
##print ("Type          Redshift      Rel. Prob.")
##print(bestForEachType)
##for i in range(10):#ntypes):
##    bestIndex = bestForEachType[i][0]
##    name, age = typeNamesList[bestIndex].split(': ')
##    print "".join(word.ljust(15) for word in [name, age , str(bestForEachType[i][1]), str(bestForEachType[i][2])])
##    #print(typeNamesList[bestIndex] + '\t' + str(bestForEachType[i][1]) + '\t' + str(bestForEachType[i][2]))
##
##
##
###Plot Each Best Type at corresponding best redshift
##for j in range(2):#len(bestForEachType)): #[0:2] takes top 2 entries
##    c = int(bestForEachType[:,0][j])
##    typeName = typeNamesList[c]
##    for i in range(0,len(trainImages)):
##        if (trainLabels[i][c] == 1):
##            print(i)
##            plt.plot(trainImages[i])
##            plt.plot(inputImages[redshiftIndex[c]])
##            plt.title(typeName+ ": " + str(bestForEachType[j][1]))
##            plt.show()
##            break
##        
##
###Plot Probability vs redshift for each class
##redshiftGraphs = [[[],[]] for i in range(ntypes)]
##for j in range(2):#len(bestForEachType)): #[0:2] takes top 2 entries
##    c = int(bestForEachType[:,0][j])
##    typeName = typeNamesList[c]
##    redshiftGraphs[c][0] = inputRedshifts
##    redshiftGraphs[c][1] = yInputRedshift[:,c]
##    plt.plot(redshiftGraphs[c][0],redshiftGraphs[c][1])
##    plt.xlabel("z")
##    plt.ylabel("Probability")
##    bestIndex = bestForEachType[c][0]
##    plt.title("Type: " + typeNamesList[bestIndex])
##    plt.show()
##
##redshiftGraphs = np.array(redshiftGraphs)
