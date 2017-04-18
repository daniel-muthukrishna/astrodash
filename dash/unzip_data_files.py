import zipfile
import os


def unzip_data_files(dataZipFilename):
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    zipRef = zipfile.ZipFile(dataZipFilename, 'r')
    zipRef.extractall(os.path.join(scriptDirectory))  # #
    zipRef.close()


if __name__ == '__main__':
    unzip_data_files('model_v01.zip')