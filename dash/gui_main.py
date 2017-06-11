import os
import sys
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from dash.design import Ui_MainWindow
from dash.restore_model import LoadInputSpectra, BestTypesListSingleRedshift, get_training_parameters, classification_split
from dash.create_arrays import AgeBinning
from dash.read_binned_templates import load_templates, get_templates, ReadBinnedTemplates
from dash.false_positive_rejection import combined_prob
from dash.calculate_redshift import get_median_redshift, get_redshift_axis


class MainApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None, inputFilename="DefaultFilename", data_files='models_v01'):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)

        self.mainDirectory = os.path.dirname(os.path.abspath(__file__))
        self.data_files = data_files
        self.templates()
        self.plotted = False
        self.plotZ = 0
        self.hostFraction = 0
        self.inputFluxes = np.zeros((2, int(self.nw)))
        self.inputImageUnRedshifted = np.zeros(int(self.nw))
        self.templatePlotFlux = np.zeros(int(self.nw))
        self.templateSubIndex = 0
        self.snName = 'Ia-norm'
        self.snAge = '-20 to -18'
        self.hostName = 'No Host'

        self.pushButtonLeftTemplate.clicked.connect(self.select_sub_template_left)
        self.pushButtonRightTemplate.clicked.connect(self.select_sub_template_right)
        self.btnBrowse.clicked.connect(self.select_input_file)
        self.listWidget.itemClicked.connect(self.list_item_clicked)
        self.btnRefit.clicked.connect(self.fit_spectra)
        self.inputFilename = inputFilename
        self.progressBar.setValue(100)
        self.add_combo_box_entries()

        self.select_tensorflow_model()
        self.checkBoxKnownZ.stateChanged.connect(self.select_tensorflow_model)
        self.checkBoxClassifyHost.stateChanged.connect(self.select_tensorflow_model)

        self.horizontalSliderSmooth.valueChanged.connect(self.smooth_slider_changed)
        self.lineEditSmooth.textChanged.connect(self.smooth_text_changed)

        self.horizontalSliderRedshift.valueChanged.connect(self.redshift_slider_changed)
        self.lineEditRedshift.textChanged.connect(self.redshift_text_changed)

        self.horizontalSliderHostFraction.valueChanged.connect(self.host_fraction_slider_changed)
        self.lineEditHostFraction.editingFinished.connect(self.host_fraction_text_changed)

        self.comboBoxSNType.currentIndexChanged.connect(self.combo_box_changed)
        self.comboBoxAge.currentIndexChanged.connect(self.combo_box_changed)
        self.comboBoxHost.currentIndexChanged.connect(self.combo_box_changed)

    def select_tensorflow_model(self):
        if self.checkBoxKnownZ.isChecked():
            self.lineEditKnownZ.setEnabled(True)
            if self.checkBoxClassifyHost.isChecked():
                self.modelFilename = os.path.join(self.mainDirectory, self.data_files, "models/zeroZ_classifyHost/tensorflow_model.ckpt")
            else:
                self.modelFilename = os.path.join(self.mainDirectory, self.data_files, "models/zeroZ/tensorflow_model.ckpt")
        else:
            self.lineEditKnownZ.setEnabled(False)
            if self.checkBoxClassifyHost.isChecked():
                self.modelFilename = os.path.join(self.mainDirectory, self.data_files, "models/agnosticZ_classifyHost/tensorflow_model.ckpt")
            else:
                self.modelFilename = os.path.join(self.mainDirectory, self.data_files, "models/agnosticZ/tensorflow_model.ckpt")
        if not os.path.isfile(self.modelFilename + ".index"):
            QtGui.QMessageBox.critical(self, "Error", "Model does not exist")

    def templates(self):
        pars = get_training_parameters()
        self.w0, self.w1, self.minAge, self.maxAge, self.ageBinSize, self.typeList, self.nTypes, self.nw, self.hostTypes \
            = pars['w0'], pars['w1'], pars['minAge'], pars['maxAge'], pars['ageBinSize'], pars['typeList'], pars['nTypes'], pars['nw'], pars['galTypeList']

        self.dwlog = np.log(self.w1/self.w0)/self.nw
        self.wave = self.w0 * np.exp(np.arange(0,self.nw) * self.dwlog)

        self.snTemplates, self.galTemplates = load_templates(os.path.join(self.mainDirectory, self.data_files, 'models/sn_and_host_templates.npz'))

    def get_sn_and_host_templates(self, snName, snAge, hostName):
        snInfos, snNames, hostInfos, hostNames = get_templates(snName, snAge, hostName, self.snTemplates, self.galTemplates, self.nw)

        return snInfos, snNames, hostInfos, hostNames

    def get_template_info(self): #
        snInfos, snNames, hostInfos, hostNames = self.get_sn_and_host_templates(self.snName, self.snAge, self.hostName)
        numOfSubTemplates = len(snNames)
        if self.templateSubIndex >= numOfSubTemplates:
            self.templateSubIndex = 0
        if self.templateSubIndex < 0:
            self.templateSubIndex = numOfSubTemplates - 1

        if snInfos != []:
            readBinnedTemplates = ReadBinnedTemplates(snInfos[self.templateSubIndex], hostInfos[0], self.w0, self.w1, self.nw)
            name = "%s_%s" % (snNames[self.templateSubIndex], hostNames[0])
            if self.hostName != "No Host":
                wave, flux = readBinnedTemplates.template_data(snCoeff=1 - self.hostFraction/100., galCoeff=self.hostFraction/100., z=0)
            else:
                wave, flux = readBinnedTemplates.template_data(snCoeff=1, galCoeff=0, z=0)
            return flux, name
        else:
            flux = np.zeros(self.nw)
            name = "NO_TEMPLATES!"

        return flux, name

    def select_sub_template_right(self):
        self.templateSubIndex += 1
        self.plot_sub_templates()

    def select_sub_template_left(self):
        self.templateSubIndex -= 1
        self.plot_sub_templates()

    def plot_sub_templates(self):
        flux, name = self.get_template_info() #

        self.templatePlotFlux = flux
        self.templatePlotName = name
        print(self.templatePlotName)
        self.plot_best_matches()
        self.plot_cross_corr(self.snName, self.snAge)

    def combo_box_changed(self):
        self.snName = str(self.comboBoxSNType.currentText())
        self.snAge = str(self.comboBoxAge.currentText())
        self.hostName = str(self.comboBoxHost.currentText())

        flux, name = self.get_template_info()
        self.templatePlotFlux = flux
        self.templatePlotName = name
        self.plot_cross_corr(self.snName, self.snAge)
        if self.knownRedshift:
            self.plot_best_matches()
        else:
            redshift, crossCorr = self.calc_redshift(self.snName, self.snAge)
            self.set_plot_redshift(redshift)
        print(self.templatePlotName)

    def add_combo_box_entries(self):
        ageLabels = AgeBinning(self.minAge, self.maxAge, self.ageBinSize).age_labels()
        for i in range(len(ageLabels)):
            self.comboBoxAge.addItem(ageLabels[i])

        for typeName in self.typeList:
            self.comboBoxSNType.addItem(typeName)

        self.comboBoxHost.addItem("No Host")
        for hostName in self.hostTypes:
            self.comboBoxHost.addItem(hostName)

    def host_fraction_slider_changed(self):
        self.hostFraction = self.horizontalSliderHostFraction.value()
        self.lineEditHostFraction.setText("%s%%" % str(self.hostFraction))
        self.templatePlotFlux, self.templatePlotName = self.get_template_info()
        self.plot_best_matches()

    def host_fraction_text_changed(self):
        try:
            self.hostFraction = float(self.lineEditHostFraction.text().strip("%%"))
            self.horizontalSliderHostFraction.setValue(int(self.hostFraction))
        except ValueError:
            print("Host Fraction Value Error")

    def redshift_slider_changed(self):
        self.plotZ = self.horizontalSliderRedshift.value()/10000.
        self.lineEditRedshift.setText(str(self.plotZ))

    def redshift_text_changed(self):
        try:
            self.plotZ = float(self.lineEditRedshift.text())
            self.horizontalSliderRedshift.setValue(int(self.plotZ*10000))
            self.plot_best_matches()
        except ValueError:
            print("Redshift Value Error")

    def set_plot_redshift(self, plotZ):
        plotZ = float(plotZ)
        self.plotZ = plotZ
        self.lineEditRedshift.setText(str(plotZ))
        self.horizontalSliderRedshift.setValue(int(plotZ*10000))
        self.plot_best_matches()

    def smooth_slider_changed(self):
        self.lineEditSmooth.setText(str(self.horizontalSliderSmooth.value()))

    def smooth_text_changed(self):
        try:
            self.horizontalSliderSmooth.setValue(int(self.lineEditSmooth.text()))
        except ValueError:
            pass

    def select_input_file(self):
        inputFilename = QtGui.QFileDialog.getOpenFileName(self, "Select a spectrum file")[0]
        print(inputFilename)
        print(self.inputFilename)
        if (inputFilename == self.inputFilename) or (inputFilename == ""):
            pass
        else:
            self.inputFilename = inputFilename
            self.lblInputFilename.setText(inputFilename.split('/')[-1])

            self.fit_spectra()

    def fit_spectra(self):
        self.cancelledFitting = False
        try:
            self.smooth = int(self.lineEditSmooth.text())
        except ValueError:
            QtGui.QMessageBox.critical(self, "Error", "Smooth must be positive integer")
            return
        try:
            self.minWave = int(self.lineEditMinWave.text())
            self.maxWave = int(self.lineEditMaxWave.text())
        except ValueError:
            QtGui.QMessageBox.critical(self, "Error", "Min and max waves must be integers between %d and %d" % (self.w0, self.w1))
            return
        if self.checkBoxClassifyHost.isChecked():
            self.classifyHost = True
        else:
            self.classifyHost = False
        if self.checkBoxKnownZ.isChecked():
            self.knownRedshift = True
            try:
                knownZ = float(self.lineEditKnownZ.text())
                self.bestRedshift = knownZ
            except ValueError:
                QtGui.QMessageBox.critical(self, "Error", "Enter Known Redshift")
                return
        else:
            self.knownRedshift = False
            knownZ = 0
            self.lineEditKnownZ.setText("")
        if not os.path.isfile(self.modelFilename + ".index"):
            QtGui.QMessageBox.critical(self, "Error", "Model does not exist")
            return

        self.progressBar.setValue(36)
        self.set_plot_redshift(knownZ)

        self.fitThread = FitSpectrumThread(self.inputFilename, knownZ, self.modelFilename, self.smooth, self.classifyHost, self.minWave, self.maxWave)
        self.fitThread.trigger.connect(self.load_spectrum_single_redshift)

        self.fitThread.start()

        self.btnCancel.clicked.connect(self.cancel)

    def cancel(self):
        if not self.cancelledFitting:
            self.cancelledFitting = True
            self.fitThread.terminate()
            self.progressBar.setValue(100)
            QtGui.QMessageBox.information(self, "Cancelled!", "Stopped Fitting Input Spectrum")

    def load_spectrum_single_redshift(self, spectrumInfo):
        self.bestTypes, self.softmax, self.idx, self.typeNamesList, self.inputImageUnRedshifted = spectrumInfo
        self.progressBar.setValue(85)#self.progressBar.value()+)
        self.done_fit_thread_single_redshift()

    def done_fit_thread_single_redshift(self):
        if not self.cancelledFitting:
            self.plotted = True
            self.list_best_matches_single_redshift()
            self.set_plot_redshift(self.bestRedshift)
            self.plot_cross_corr(self.snName, self.snAge)
            self.progressBar.setValue(100)
            QtGui.QMessageBox.information(self, "Done!", "Finished Fitting Input Spectrum")

    def best_broad_type(self):
        bestMatchList = []
        for i in range(10):
            host, name, age = classification_split(self.bestTypes[i])
            bestMatchList.append([host, name, age, self.softmax[i]])
        host, prevName, bestAge, probTotal, reliableFlag = combined_prob(bestMatchList)
        self.labelBestSnType.setText(prevName)
        self.labelBestAgeRange.setText(bestAge)
        self.labelBestHostType.setText(host)
        self.labelBestRedshift.setText(str(self.bestRedshift))
        self.labelBestRelProb.setText("%s%%" % str(round(100*probTotal, 2)))
        if reliableFlag:
            self.labelReliableFlag.setText("Reliable")
            self.labelReliableFlag.setStyleSheet('color: green')
        else:
            self.labelReliableFlag.setText("Unreliable")
            self.labelReliableFlag.setStyleSheet('color: red')

    def list_best_matches_single_redshift(self):
        print("listing best matches...")
        redshifts = self.best_redshifts()
        self.listWidget.clear()
        if self.knownRedshift:
            self.listWidget.addItem("".join(word.ljust(25) for word in ['No.', 'Type', 'Age', 'Softmax Prob.']))
        else:
            self.listWidget.addItem("".join(word.ljust(25) for word in ['No.', 'Type', 'Age', 'Redshift', 'Softmax Prob.']))
        for i in range(20):
            host, name, age = classification_split(self.bestTypes[i])
            prob = self.softmax[i]
            redshift = redshifts[i]
            if self.classifyHost:
                if self.knownRedshift:
                    self.listWidget.addItem("".join(word.ljust(25) for word in [str(i + 1), host, name, age, str(prob)]))
                else:
                    self.listWidget.addItem("".join(word.ljust(25) for word in [str(i + 1), host, name, age, str(redshift), str(prob)]))
            else:
                if self.knownRedshift:
                    self.listWidget.addItem("".join(word.ljust(25) for word in [str(i + 1), name, age, str(prob)]))
                else:
                    self.listWidget.addItem("".join(word.ljust(25) for word in [str(i + 1), name, age, str(redshift), str(prob)]))


            if i == 0:
                SNTypeComboBoxIndex = self.comboBoxSNType.findText(name)
                self.comboBoxSNType.setCurrentIndex(SNTypeComboBoxIndex)
                AgeComboBoxIndex = self.comboBoxAge.findText(age)
                self.comboBoxAge.setCurrentIndex(AgeComboBoxIndex)
                hostComboBoxIndex = self.comboBoxHost.findText(host)
                self.comboBoxHost.setCurrentIndex(hostComboBoxIndex)
            if not self.knownRedshift:
                self.bestRedshift = redshifts[0]
        self.best_broad_type()

    def list_item_clicked(self, item):
        if item.text()[0].isdigit():
            self.templateSubIndex = 0
            host = "No Host"
            if self.knownRedshift:
                if self.classifyHost:
                    index, host, snTypePlot, age1, to, age3, softmax = str(item.text()).split()
                else:
                    index, snTypePlot, age1, to, age3, softmax = str(item.text()).split()
            else:
                if self.classifyHost:
                    index, host, self.snTypePlot, age1, to, age3, redshift, softmax = str(item.text()).split()
                else:
                    index, snTypePlot, age1, to, age3, redshift, softmax = str(item.text()).split()
                self.set_plot_redshift(redshift)
            agePlot = age1 + ' to ' + age3

            self.plot_cross_corr(self.snName, self.snAge)
            snTypeComboBoxIndex = self.comboBoxSNType.findText(snTypePlot)
            self.comboBoxSNType.setCurrentIndex(snTypeComboBoxIndex)
            AgeComboBoxIndex = self.comboBoxAge.findText(agePlot)
            self.comboBoxAge.setCurrentIndex(AgeComboBoxIndex)
            hostComboBoxIndex = self.comboBoxHost.findText(host)
            self.comboBoxHost.setCurrentIndex(hostComboBoxIndex)

    def plot_best_matches(self):
        if self.plotted:
            templateWave = self.wave * (1 + (self.plotZ))
            self.labelTemplateName.setText(self.templatePlotName)

            self.graphicsView.clear()
            inputPlotFlux = self.inputImageUnRedshifted
            self.graphicsView.plot(self.wave, inputPlotFlux, name='Input Spectrum', pen={'color': (0, 255, 0)})
            self.graphicsView.plot(templateWave, self.templatePlotFlux, name=self.templatePlotName, pen={'color': (255,0,0)})
            self.graphicsView.setXRange(2500, 10000)
            self.graphicsView.setYRange(0, 1)
            self.graphicsView.plotItem.showGrid(x=True, y=True, alpha=0.95)

    def best_redshifts(self):
        redshifts = []
        for i in range(20):
            host, name, age = classification_split(self.bestTypes[i])
            redshift, crossCorr = self.calc_redshift(name, age)
            redshifts.append(redshift)
        return redshifts

    def calc_redshift(self, snName, snAge):
        host = "No Host"
        snInfos, snNames, hostInfos, hostNames = self.get_sn_and_host_templates(snName, snAge, host)
        numOfSubTemplates = len(snNames)
        templateFluxes = []
        for i in range(numOfSubTemplates):
            templateFluxes.append(snInfos[i][1])

        redshift, crossCorr = get_median_redshift(self.inputImageUnRedshifted, templateFluxes, self.nw, self.dwlog)
        print(redshift)
        if redshift is None:
            return 0, np.zeros(1024)

        return round(redshift, 4), crossCorr

    def plot_cross_corr(self, snName, snAge):
        zAxis = get_redshift_axis(self.nw, self.dwlog)
        redshift, crossCorr = self.calc_redshift(snName, snAge)
        self.graphicsView_2.clear()
        self.graphicsView_2.plot(zAxis, crossCorr)
        self.graphicsView_2.setXRange(0, 1)
        self.graphicsView_2.setYRange(min(crossCorr), max(crossCorr))
        self.graphicsView_2.plotItem.showGrid(x=True, y=True, alpha=0.95)
        # self.graphicsView_2.plotItem.setLabels(bottom="z")

    def browse_folder(self):
        self.listWidget.clear()
        directory = QtGui.QFileDialog.getExistingDirectory(self, "Pick a folder")

        if directory:
            for file_name in os.listdir(directory):
                self.listWidget.addItem(file_name)


class FitSpectrumThread(QThread):

    trigger = pyqtSignal('PyQt_PyObject')

    def __init__(self, inputFilename, knownZ, modelFilename, smooth, classifyHost, minWave, maxWave):
        QThread.__init__(self)
        self.inputFilename = str(inputFilename)
        self.knownZ = knownZ
        self.modelFilename = modelFilename
        self.smooth = smooth
        self.classifyHost = classifyHost
        self.minWave = minWave
        self.maxWave = maxWave

    def __del__(self):
        self.wait()

    def _input_spectrum_single_redshift(self):
        trainParams = get_training_parameters()
        loadInputSpectraUnRedshifted = LoadInputSpectra(self.inputFilename, 0, 0, self.smooth, trainParams, self.minWave, self.maxWave, self.classifyHost)
        inputImageUnRedshifted, inputRedshift, typeNamesList, nw, nBins = loadInputSpectraUnRedshifted.input_spectra()

        loadInputSpectra = LoadInputSpectra(self.inputFilename, self.knownZ, self.knownZ, self.smooth, trainParams, self.minWave, self.maxWave, self.classifyHost)
        inputImage, inputRedshift, typeNamesList, nw, nBins = loadInputSpectra.input_spectra()
        bestTypesList = BestTypesListSingleRedshift(self.modelFilename, inputImage, typeNamesList, nw, nBins)
        bestTypes = bestTypesList.bestTypes[0]
        softmax = bestTypesList.softmaxOrdered[0]
        idx = bestTypesList.idx[0]

        return bestTypes, softmax, idx, typeNamesList, inputImageUnRedshifted[0]

    def run(self):
        spectrumInfo = self._input_spectrum_single_redshift()
        self.trigger.emit(spectrumInfo)


def main():
    app = QtGui.QApplication(sys.argv)
    form = MainApp()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()
