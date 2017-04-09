from dash.training_parameters import create_training_params_file
from dash.create_training_set import create_training_set_files
from dash.create_template_set import create_template_set_file
from dash.deep_learning_multilayer import train_model
import zipfile
import os
import shutil

if __name__ == '__main__':
    filenames = []

    # CREATE PARAMETERS PICKLE FILE
    trainingParamsFilename = create_training_params_file()
    filenames.append(trainingParamsFilename)

    # CREATE TRAINING SET FILES
    trainingSetFilename = create_training_set_files()
    filenames.append(trainingSetFilename)

    # CREATE TEMPLATE SET FILE
    templateSetFilename = create_template_set_file()
    filenames.append(templateSetFilename)

    # TRAIN TENSORFLOW MODEL
    modelFilenames = train_model()
    filenames.extend(modelFilenames)

    # SAVE ALL FILES TO ZIP FILE
    dataFilesZip = 'data_files_v01.zip'
    with zipfile.ZipFile(dataFilesZip, 'w') as myzip:
        for f in filenames:
            myzip.write(f)

    # Delete data_files folder since they are now in the zip files
    for filename in filenames:
        os.remove(filename)
