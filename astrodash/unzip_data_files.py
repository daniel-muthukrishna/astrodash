import zipfile
import os


def unzip_data_files(dataZipFilename):
    print("Unzipping data files...")
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    zipRef = zipfile.ZipFile(os.path.join(scriptDirectory, dataZipFilename), 'r')
    zipRef.extractall(os.path.join(scriptDirectory, dataZipFilename.strip(".zip")))
    zipRef.close()
    os.remove(os.path.join(scriptDirectory, dataZipFilename))
    print("Data files installed!")


if __name__ == '__main__':
    unzip_data_files('models_v06.zip')
