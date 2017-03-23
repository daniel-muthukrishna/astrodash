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

    oldFilenames = ['type_age_atRedshiftZero.npz', 'training_params.pickle', 'templates.npz']
    delete_previous_versions(oldFilenames, scriptDirectory)

    saveFilenames = ['model_trainedAtZeroZ.ckpt', 'type_age_atRedshiftZero_v02.npz', 'training_params_v02.pickle',
                     'templates_v02.npz']

    urlpaths = ["https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/model_trainedAtZeroZ.ckpt",
                "https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/type_age_atRedshiftZero_v02.npz",
                "https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/training_params_v02.pickle",
                "https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/templates_v02.npz"]

    printStatuses = ["Downloading Trained Model...", "Downloading Training Data files...",
                     "Downloading Model Parameters File...", "Downloading Template Data files..."]

    for i in range(len(urlpaths)):
        download_file(saveFilenames[i], urlpaths[i], printStatuses[i], scriptDirectory)


def delete_previous_versions(oldFilenames, scriptDirectory):
    for oldFilename in oldFilenames:
        dataFilename = os.path.join(scriptDirectory, oldFilename)
        if os.path.isfile(dataFilename):
            print("Deleting previous version of data file: %s" % oldFilename)
            os.remove(dataFilename)

