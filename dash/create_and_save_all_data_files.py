from dash.training_parameters import create_training_params_file
from dash.create_training_set import create_training_set_files
from dash.create_template_set import create_template_set_file
from dash.deep_learning_multilayer import train_model
import zipfile
import os
import shutil
import time

if __name__ == '__main__':
    dataDirName = 'data_files/'
    dataFilenames = []
    if not os.path.exists(dataDirName):
        os.makedirs(dataDirName)

    # CREATE PARAMETERS PICKLE FILE
    t1 = time.time()
    trainingParamsFilename = "data_files/training_params.pickle"  # create_training_params_file(dataDirName)
    dataFilenames.append(trainingParamsFilename)
    t2 = time.time()
    print("time spent: {0:.2f}".format(t2 - t1))

    # CREATE TRAINING SET FILES
    trainingSetFilename = "data_files/training_set.zip"  # create_training_set_files(dataDirName, minZ=0, maxZ=0., redshiftPrecision=0.1, trainWithHost=True, classifyHost=False)
    dataFilenames.append(trainingSetFilename)
    t3 = time.time()
    print("time spent: {0:.2f}".format(t3 - t2))

    # CREATE TEMPLATE SET FILE
    templateSetFilename = "data_files/templates.npz"  # create_template_set_file(dataDirName)
    dataFilenames.append(templateSetFilename)
    t4 = time.time()
    print("time spent: {0:.2f}".format(t4 - t3))

    # TRAIN TENSORFLOW MODEL
    modelFilenames = train_model()
    dataFilenames.extend(modelFilenames)
    t5 = time.time()
    print("time spent: {0:.2f}".format(t5 - t4))

    # SAVE ALL FILES TO ZIP FILE
    dataFilesZip = 'data_files_withHost_350kTrain100_v01.zip'
    with zipfile.ZipFile(dataFilesZip, 'w') as myzip:
        for f in dataFilenames:
            myzip.write(f)

    modelZip = 'model__withHost_350kTrain100_v01.zip'
    with zipfile.ZipFile(modelZip, 'w') as myzip:
        for f in [dataFilenames[0]] + dataFilenames[2:]:
            myzip.write(f)

    # Delete data_files folder since they are now in the zip files
    # for filename in filenames:
    #     os.remove(filename)
