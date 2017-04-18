import os
import sys
import shutil
from dash.unzip_data_files import unzip_data_files


def download_file(filename, urlpath, printStatus, scriptDirectory, zipVersion):
    dataFilename = os.path.join(scriptDirectory, filename)
    if not os.path.isfile(dataFilename):
        print(printStatus)
        if sys.version_info[0] < 3:
            import urllib
            dataFileDownload = urllib.URLopener()
            dataFileDownload.retrieve(urlpath, dataFilename)
        else:
            import urllib.request
            urllib.request.urlretrieve(urlpath, dataFilename)

        print(dataFilename)
        unzip_data_files('model_{0}.zip'.format(zipVersion))


def delete_previous_versions(oldFilenames, scriptDirectory):
    for oldFilename in oldFilenames:
        dataFilename = os.path.join(scriptDirectory, oldFilename)
        if os.path.isfile(dataFilename):
            print("Deleting previous version of data file: %s" % oldFilename)
            os.remove(dataFilename)
            shutil.rmtree(os.path.join(scriptDirectory, 'data_files'))


def download_all_files(zipVersion):
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))

    oldFilenames = ['model_trainedAtZeroZ.ckpt', 'type_age_atRedshiftZero.npz', 'training_params.pickle', 'templates.npz',
                    'model_trainedAtZeroZ_v02.ckpt', 'type_age_atRedshiftZero_v02.npz', 'training_params_v02.pickle', 'templates_v02.npz',
                    'model_trainedAtZeroZ_v03.ckpt', 'type_age_atRedshiftZero_v03.npz', 'training_params.pickle', 'templates_v03.npz',
                    'model_trainedAtZeroZ_v03.ckpt.data-00000-of-00001', 'model_trainedAtZeroZ_v03.ckpt.index', 'model_trainedAtZeroZ_v03.ckpt.meta',
                    'model_v00.zip']
    delete_previous_versions(oldFilenames, scriptDirectory)

    saveFilenames = ['model_{0}.zip'.format(zipVersion)]

    urlpaths = ["https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/model_{0}.zip".format(zipVersion)]

    printStatuses = ["Downloading data files..."]

    for i in range(len(urlpaths)):
        download_file(saveFilenames[i], urlpaths[i], printStatuses[i], scriptDirectory, zipVersion)


if __name__ == '__main__':
    download_all_files('v01')



