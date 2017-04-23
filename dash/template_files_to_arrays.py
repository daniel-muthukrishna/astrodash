from dash.preprocessing import ReadSpectrumFile
from dash.create_arrays import TempList
import pickle
import os
import gzip


class SaveTemplateSpectra(object):
    def __init__(self, parameterFile):
        with open(parameterFile, 'rb') as f:
            pars = pickle.load(f)
            self.w0, self.w1, self.nw = pars['w0'], pars['w1'], pars['nw']

    def read_template_file(self, filename):
        readSpectrumFile = ReadSpectrumFile(filename, self.w0, self.w1, self.nw)
        spectrum = readSpectrumFile.file_extension()

        return spectrum

    def template_spectra_to_list(self, tempFileList, templateDirectory):
        tempList = TempList().temp_list(tempFileList)
        templates = []
        for filename in tempList:
            spectrum = self.read_template_file(templateDirectory+filename)
            templates.append(spectrum)
            print(filename)

        return templates

    def save_templates(self, snTempFileList, snTemplateDirectory, galTempFileList, galTemplateDirectory, saveFilename):
        snTemplates = self.template_spectra_to_list(snTempFileList, snTemplateDirectory)
        galTemplates = self.template_spectra_to_list(galTempFileList, galTemplateDirectory)
        templates = {'sn': snTemplates, 'gal': galTemplates}

        # Saving the objects
        with gzip.open(saveFilename, 'wb') as f:
            pickle.dump(templates, f, protocol=2)
        print("Saved templates to %s" % saveFilename)


def save_templates():
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))

    snidTemplateDirectory = os.path.join(scriptDirectory, "../templates/snid_templates_Modjaz_BSNIP/")
    snidTempFileList = snidTemplateDirectory + 'templist.txt'
    galTemplateDirectory = os.path.join(scriptDirectory, "../templates/superfit_templates/gal/")
    galTempFileList = galTemplateDirectory + 'gal.list'

    saveTemplateSpectra = SaveTemplateSpectra('data_files/training_params.pickle')
    saveFilename = 'data_files/sn_and_gal_templates.pklz'
    saveTemplateSpectra.save_templates(snidTempFileList, snidTemplateDirectory, galTempFileList, galTemplateDirectory, saveFilename)

    return saveFilename


if __name__ == '__main__':
    templatesFilename1 = save_templates()
