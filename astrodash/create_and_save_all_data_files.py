from astrodash.training_parameters import create_training_params_file
from astrodash.create_training_set import create_training_set_files
from astrodash.create_template_set import create_template_set_file
from astrodash.deep_learning_multilayer import train_model
import zipfile
import os
import glob
import shutil
import time

scriptDirectory = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    modelName = 'testingZeroZ'
    trainWithHost = True
    classifyHost = False
    minZ = 0.
    maxZ = 0.
    redshiftDuringTraining = True
    trainFraction = 0.8
    numTrainBatches = 2000000
    # Do not change this unless we want to redshift before training.
    numOfRedshifts = 1

    if numOfRedshifts != 1:
        redshiftDuringTraining = False

    dataDirName = os.path.join(scriptDirectory, 'data_files_{0}/'.format(modelName))
    dataFilenames = []
    if not os.path.exists(dataDirName):
        os.makedirs(dataDirName)

    # MAKE INFO TEXT FILE ABOUT MODEL
    modelInfoFilename = dataDirName + "model_info.txt"
    with open(modelInfoFilename, "w") as f:
        f.write("Date Time: %s\n" % time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        f.write("Directory: %s\n" % dataDirName)
        f.write("Add Host: {}\n".format(trainWithHost))
        f.write("SN-Host fractions: [0.99, 0.98, 0.95, 0.93, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]\n")
        f.write("Classify Host: {}\n".format(classifyHost))
        f.write("Redshift: Zero\n")
        f.write("Redshift Range: {} to {}\n".format(minZ, maxZ))
        f.write("Num of Redshifts: {}\n".format(numOfRedshifts))
        f.write("Redshift During Training: {}\n".format(redshiftDuringTraining))
        f.write("Fraction of Training Set Used: {}\n".format(trainFraction))
        f.write("Training Amount: 50 x {}\n".format(numTrainBatches))
        f.write("Changed wavelength range to 3500 to 10000A\n")
        f.write("Set outer region to 0.5\n")
        f.write("Using 2 convolutional layers in neural network\n")
        dataFilenames.append(modelInfoFilename)

    # CREATE PARAMETERS PICKLE FILE
    t1 = time.time()
    trainingParamsFilename = create_training_params_file(dataDirName)  # os.path.join(dataDirName, 'training_params.pickle')
    dataFilenames.append(trainingParamsFilename)
    t2 = time.time()
    print("time spent: {0:.2f}".format(t2 - t1))

    # CREATE TRAINING SET FILES
    trainingSetFilename = create_training_set_files(dataDirName, minZ=minZ, maxZ=maxZ, numOfRedshifts=numOfRedshifts, trainWithHost=trainWithHost, classifyHost=classifyHost, trainFraction=trainFraction)  # os.path.join(dataDirName, 'training_set.zip')
    dataFilenames.append(trainingSetFilename)
    t3 = time.time()
    print("time spent: {0:.2f}".format(t3 - t2))

    # TRAIN TENSORFLOW MODEL
    modelFilenames = train_model(dataDirName, overwrite=True, numTrainBatches=numTrainBatches, minZ=minZ, maxZ=maxZ, redshifting=redshiftDuringTraining)
    dataFilenames.extend(modelFilenames)
    t4 = time.time()
    print("time spent: {0:.2f}".format(t4 - t3))

    # SAVE ALL FILES TO ZIP FILE
    dataFilesZip = 'data_files_{0}.zip'.format(modelName)
    with zipfile.ZipFile(dataFilesZip, 'w') as myzip:
        for f in dataFilenames:
            myzip.write(f)

    modelZip = 'model_{0}.zip'.format(modelName)
    with zipfile.ZipFile(modelZip, 'w') as myzip:
        for f in dataFilenames[0:2] + dataFilenames[3:]:
            myzip.write(f)

    # # Delete temporary memory mapping files
    # for filename in glob.glob('shuffled*.dat'):
    #     os.remove(filename)
    # for filename in glob.glob('oversampled*.dat'):
    #     os.remove(filename)

    # Delete data_files folder since they are now in the zip files
    # for filename in filenames:
    #     os.remove(filename)
