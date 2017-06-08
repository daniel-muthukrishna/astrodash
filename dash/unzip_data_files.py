import zipfile
import os


def unzip_data_files(dataZipFilename):
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    zipRef = zipfile.ZipFile(os.path.join(scriptDirectory, dataZipFilename, 'r'))
    zipRef.extractall(os.path.join(scriptDirectory, dataZipFilename.strip(".zip")))
    zipRef.close()
    os.remove(os.path.join(scriptDirectory, dataZipFilename))


if __name__ == '__main__':
    unzip_data_files('models_v01.zip')