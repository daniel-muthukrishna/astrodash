from dash.training_parameters import create_training_params_file
from dash.create_training_set import create_training_set_files
from dash.create_template_set import create_template_set_file
from dash.deep_learning_multilayer import train_model
import zipfile
import os
import shutil

if __name__ == '__main__':
    dataFilenames = []

    # CREATE PARAMETERS PICKLE FILE
    trainingParamsFilename = 'data_files/trainingSet_type_age_atRedshiftZero' # create_training_params_file()
    dataFilenames.append(trainingParamsFilename)

    # CREATE TRAINING SET FILES
    trainingSetFilename = 'data_files/trainingSet_type_age_atRedshiftZero.zip' # create_training_set_files()
    dataFilenames.append(trainingSetFilename)

    # CREATE TEMPLATE SET FILE
    templateSetFilename = 'data_files/templates.npz' # create_template_set_file()
    dataFilenames.append(templateSetFilename)

    # TRAIN TENSORFLOW MODEL
    modelFilenames = train_model()
    dataFilenames.extend(modelFilenames)

    # SAVE ALL FILES TO ZIP FILE
    dataFilesZip = 'data_files_withVaryingHost_v01.zip'
    with zipfile.ZipFile(dataFilesZip, 'w') as myzip:
        for f in dataFilenames:
            myzip.write(f)

    modelZip = 'model_withVaryingHost_v01.zip'
    with zipfile.ZipFile(modelZip, 'w') as myzip:
        for f in [dataFilenames[0]] + dataFilenames[2:]:
            myzip.write(f)

    # Delete data_files folder since they are now in the zip files
    # for filename in filenames:
    #     os.remove(filename)
