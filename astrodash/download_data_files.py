import os
import sys
import shutil
from astrodash.unzip_data_files import unzip_data_files


def download_file(filename, urlpath, printStatus, scriptDirectory):
    dataFilename = os.path.join(scriptDirectory, filename)
    if (not os.path.isfile(dataFilename)) and (not os.path.isdir(dataFilename.strip(".zip"))):
        print(printStatus)
        if sys.version_info[0] < 3:
            import urllib
            urllib.urlretrieve(urlpath, dataFilename)
        else:
            import urllib.request
            urllib.request.urlretrieve(urlpath, dataFilename)

        print("Download complete! File saved to %s." % dataFilename)
        if filename.endswith(".zip"):
            unzip_data_files(filename)


def delete_previous_versions(oldFilenames, scriptDirectory):
    for oldFilename in oldFilenames:
        dataFilename = os.path.join(scriptDirectory, oldFilename)
        if os.path.isfile(dataFilename):
            print("Deleting previous version of data file: %s" % oldFilename)
            os.remove(dataFilename)
        if os.path.isdir(dataFilename.strip('.zip')):
            shutil.rmtree(dataFilename.strip('.zip'))


def download_all_files(zipVersion):
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))

    oldFilenames = ['models_v01.zip', 'models_v02.zip', 'models_v03.zip', 'models_v04.zip', 'models_v05.zip']

    delete_previous_versions(oldFilenames, scriptDirectory)

    saveFilenames = ['models_{0}.zip'.format(zipVersion)]

    urlpaths = ["https://github.com/daniel-muthukrishna/astrodash/blob/master/astrodash/models_{0}.zip?raw=true".format(zipVersion)]

    printStatuses = ["Downloading data files from {0}... \nThis download contains the Tensorflow models trained using deep learning. \n"
                     "The file to be downloaded is ~200MB. However, this is a one time download, and will only occur\n"
                     "the very first time that this version of DASH is installed.".format(urlpaths[0])]

    for i in range(len(urlpaths)):
        download_file(saveFilenames[i], urlpaths[i], printStatuses[i], scriptDirectory)


if __name__ == '__main__':
    download_all_files('v06')



