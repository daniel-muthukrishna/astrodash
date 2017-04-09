import zipfile


def unzip_data_files(dataZipFilename):
    extractedFolder = 'data_files'
    zipRef = zipfile.ZipFile(dataZipFilename, 'r')
    zipRef.extractall(extractedFolder)
    zipRef.close()


if __name__ == '__main__':
    unzip_data_files('data_files_v01.zip')