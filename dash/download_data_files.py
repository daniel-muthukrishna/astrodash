import os
import sys


def download_file(filename, urlpath, printStatus, scriptDirectory):
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


def download_all_files():
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))

    oldFilenames = ['model_trainedAtZeroZ.ckpt', 'type_age_atRedshiftZero.npz', 'training_params.pickle', 'templates.npz',
                    'model_trainedAtZeroZ_v02.ckpt', 'type_age_atRedshiftZero_v02.npz', 'training_params_v02.pickle', 'templates_v02.npz']
    delete_previous_versions(oldFilenames, scriptDirectory)

    saveFilenames = ['model_trainedAtZeroZ_v03.ckpt.data-00000-of-00001',
                     'model_trainedAtZeroZ_v03.ckpt.index',
                     'model_trainedAtZeroZ_v03.ckpt.meta',
                     'type_age_atRedshiftZero_v03.npz',
                     'training_params_v03.pickle',
                     'templates_v03.npz']

    urlpaths = ["https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/model_trainedAtZeroZ_v03.ckpt.data-00000-of-00001",
                "https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/model_trainedAtZeroZ_v03.ckpt.index",
                "https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/model_trainedAtZeroZ_v03.ckpt.meta",
                "https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/type_age_atRedshiftZero_v03.npz",
                "https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/training_params_v03.pickle",
                "https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/templates_v03.npz"]

    printStatuses = ["Downloading Tensorflow trained model 1 of 3...",
                     "Downloading Tensorflow trained model 2 of 3...",
                     "Downloading Tensorflow trained model 3 of 3...",
                     "Downloading template data file...",
                     "Downloading model parameters file...",
                     "Downloading template data files..."]

    for i in range(len(urlpaths)):
        download_file(saveFilenames[i], urlpaths[i], printStatuses[i], scriptDirectory)


def delete_previous_versions(oldFilenames, scriptDirectory):
    for oldFilename in oldFilenames:
        dataFilename = os.path.join(scriptDirectory, oldFilename)
        if os.path.isfile(dataFilename):
            print("Deleting previous version of data file: %s" % oldFilename)
            os.remove(dataFilename)

