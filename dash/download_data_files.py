import os
import sys
import urllib


def download_file(filename, urlPath, printStatus, scriptDirectory):
    dataFilename = os.path.join(scriptDirectory, filename)
    if not os.path.isfile(dataFilename):
        print(printStatus)
        if sys.version_info[0] < 3:
            dataFileDownload = urllib.URLopener()
            dataFileDownload.retrieve(urlPath, dataFilename)
        else:
            urllib.request.retrieve(urlPath, dataFilename)

        print(dataFilename)


def download_all_files():
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    saveFilenames = ['model_trainedAtZeroZ.ckpt', 'type_age_atRedshiftZero.npz',
                     'training_params.pickle', 'templates.npz']

    urlpaths = ["https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/model_trainedAtZeroZ.ckpt",
                "https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/type_age_atRedshiftZero.npz",
                "https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/training_params.pickle",
                "https://raw.githubusercontent.com/daniel-muthukrishna/DASH/master/dash/templates.npz"]

    printStatuses = ["Downloading Trained Model...", "Downloading Training Data files...",
                     "Downloading Model Parameters File...", "Downloading Template Data files..."]

    for i in range(len(urlpaths)):
        download_file(saveFilenames[i], urlpaths[i], printStatuses[i], scriptDirectory)
