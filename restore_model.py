import tensorflow as tf
import numpy as np
import pickle
from input_spectra import *

loaded = np.load('/home/dan/Desktop/SNClassifying_Pre-alpha/file_w_ages2.npz')
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
        self.x = tf.placeholder(tf.float32, [None, nw])
        self.W = tf.Variable(tf.zeros([nw, nBins]))
        self.b = tf.Variable(tf.zeros([nBins]))
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        self.y_ = tf.placeholder(tf.float32, [None, nBins])

        self.saver = tf.train.Saver()

    def restore_variables(self):
        with tf.Session() as sess:
            self.saver.restore(sess, self.modelFilename)
            yInputRedshift = sess.run(self.y, feed_dict={self.x: self.inputImages})
            correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(yInputRedshift)
            
        return yInputRedshift

    def reset(self):
        tf.reset_default_graph()


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
                print("No Template")
                templateFlux = np.zeros(len(trainImages[0]))
                inputFlux = self.inputImages[int(self.redshiftIndex[c])]

            templateFluxes.append(templateFlux)
            inputFluxes.append(inputFlux)
            
        templateFluxes = np.array(templateFluxes)
        inputFluxes.append(inputFluxes)
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
