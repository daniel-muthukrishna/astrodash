import os
import sys
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal
import pyqtgraph as pg
from astrodash.design import Ui_MainWindow
from astrodash.restore_model import LoadInputSpectra, BestTypesListSingleRedshift, get_training_parameters, classification_split
from astrodash.create_arrays import AgeBinning
from astrodash.read_binned_templates import load_templates, get_templates, combined_sn_and_host_data
from astrodash.false_positive_rejection import combined_prob, RlapCalc
from astrodash.calculate_redshift import get_median_redshift, get_redshift_axis
from astrodash.read_from_catalog import catalogDict


class MainApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None, inputFilename="DefaultFilename", data_files='models_v06'):
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
        self.lineEditKnownZ.setText('')
        self.infLine = pg.InfiniteLine(self.plotZ, pen={'width': 3, 'color': (135, 206, 250)}, movable=True, bounds=[-1, 1], hoverPen={'color': (255, 0, 0), 'width': 3}, label='z', labelOpts={'color': (135, 206, 250), 'position': 0.2})
        self.infLine.sigPositionChanged.connect(self.cross_corr_redshift_changed)

        self.pushButtonLeftTemplate.clicked.connect(self.select_sub_template_left)
        self.pushButtonRightTemplate.clicked.connect(self.select_sub_template_right)
        self.btnBrowse.clicked.connect(self.select_input_file)
        self.listWidget.itemClicked.connect(self.list_item_clicked)
        self.btnRefit.clicked.connect(self.fit_spectra)
        self.inputFilename = inputFilename
        self.inputFilenameSetText = inputFilename
        self.progressBar.setValue(100)
        self.add_combo_box_entries()
        self.labelRlapScore.setText('')

        self.select_tensorflow_model()
        self.checkBoxKnownZ.stateChanged.connect(self.select_tensorflow_model)
        self.checkBoxClassifyHost.stateChanged.connect(self.select_tensorflow_model)

        self.lineEditInputFilename.textChanged.connect(self.input_filename_changed)

        self.horizontalSliderSmooth.valueChanged.connect(self.smooth_slider_changed)
        self.lineEditSmooth.textChanged.connect(self.smooth_text_changed)

        self.horizontalSliderRedshift.valueChanged.connect(self.redshift_slider_changed)
        self.lineEditRedshift.textChanged.connect(self.redshift_text_changed)

        self.horizontalSliderHostFraction.valueChanged.connect(self.host_fraction_slider_changed)
        self.lineEditHostFraction.editingFinished.connect(self.host_fraction_text_changed)

        self.comboBoxSNType.currentIndexChanged.connect(self.combo_box_changed)
        self.comboBoxAge.currentIndexChanged.connect(self.combo_box_changed)
        self.comboBoxHost.currentIndexChanged.connect(self.combo_box_changed)

        self.btnQuit.clicked.connect(self.close)

        self.graphicsView.plotItem.layout.removeItem(self.graphicsView.plotItem.getAxis('top'))
        self.cAxis = pg.AxisItem(orientation='top', parent=self.graphicsView.plotItem)
        self.cAxis.linkToView(self.graphicsView.plotItem.vb)
        self.graphicsView.plotItem.axes['top']['item'] = self.cAxis
        self.graphicsView.plotItem.layout.addItem(self.cAxis, 1, 1)


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
            name = "%s_%s" % (snNames[self.templateSubIndex], hostNames[0])
            if self.hostName != "No Host":
                snCoeff = 1 - self.hostFraction/100.
                galCoeff = self.hostFraction / 100.
            else:
                snCoeff = 1
                galCoeff = 0
            wave, flux, minMaxIndex = combined_sn_and_host_data(snCoeff=snCoeff, galCoeff=galCoeff, z=0, snInfo=snInfos[self.templateSubIndex], galInfo=hostInfos[0], w0=self.w0, w1=self.w1, nw=self.nw)
            return flux, name, minMaxIndex
        else:
            flux = np.zeros(self.nw)
            name = "NO_TEMPLATES!"
            minMaxIndex = (0, 0)

        return flux, name, minMaxIndex

    def select_sub_template_right(self):
        self.templateSubIndex += 1
        self.plot_sub_templates()

    def select_sub_template_left(self):
        self.templateSubIndex -= 1
        self.plot_sub_templates()

    def plot_sub_templates(self):
        self.templatePlotFlux, self.templatePlotName, self.templateMinMaxIndex = self.get_template_info()
        print(self.templatePlotName)
        self.plot_best_matches()
        self.plot_cross_corr()

    def combo_box_changed(self):
        self.snName = str(self.comboBoxSNType.currentText())
        self.snAge = str(self.comboBoxAge.currentText())
        self.hostName = str(self.comboBoxHost.currentText())

        self.redshift, self.crossCorrs, self.medianName, self.redshiftErr = self.calc_redshift(self.snName, self.snAge)
        self.set_template_sub_index(self.medianName)
        self.templatePlotFlux, self.templatePlotName, self.templateMinMaxIndex = self.get_template_info()
        self.plot_cross_corr()
        if self.knownRedshift:
            self.plot_best_matches()
        else:
            self.set_plot_redshift(self.redshift)

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
        self.templatePlotFlux, self.templatePlotName, self.templateMinMaxIndex = self.get_template_info()
        self.plot_best_matches()

    def host_fraction_text_changed(self):
        try:
            self.hostFraction = float(self.lineEditHostFraction.text().strip("%%"))
            self.horizontalSliderHostFraction.setValue(int(self.hostFraction))
        except ValueError:
            print("Host Fraction Value Error")

    def redshift_slider_changed(self):
        self.plotZ = self.horizontalSliderRedshift.value()/10000.
        self.lineEditRedshift.setText(str(round(self.plotZ, 3)))
        self.infLine.setValue(self.plotZ)

    def redshift_text_changed(self):
        try:
            self.plotZ = float(self.lineEditRedshift.text())
            self.horizontalSliderRedshift.setValue(int(self.plotZ*10000))
            self.infLine.setValue(self.plotZ)
            self.plot_best_matches()
        except ValueError:
            print("Redshift Value Error")

    def cross_corr_redshift_changed(self):
        self.plotZ = self.infLine.value()
        self.horizontalSliderRedshift.setValue(int(self.plotZ * 10000))
        self.lineEditRedshift.setText(str(round(self.plotZ, 3)))

    def set_plot_redshift(self, plotZ):
        plotZ = float(plotZ)
        self.plotZ = plotZ
        self.lineEditRedshift.setText(str(round(self.plotZ, 3)))
        self.horizontalSliderRedshift.setValue(int(plotZ*10000))
        self.infLine.setValue(self.plotZ)
        self.plot_best_matches()

    def smooth_slider_changed(self):
        self.lineEditSmooth.setText(str(self.horizontalSliderSmooth.value()))

    def smooth_text_changed(self):
        try:
            self.horizontalSliderSmooth.setValue(int(self.lineEditSmooth.text()))
        except ValueError:
            pass

    def input_filename_changed(self):
        inputFilename = self.lineEditInputFilename.text()
        if inputFilename == self.inputFilenameSetText or inputFilename == self.inputFilename or inputFilename == "":
            pass
        else:
            self.inputFilename = inputFilename

    def select_input_file(self):
        inputFilename = QtGui.QFileDialog.getOpenFileName(self, "Select a spectrum file")[0]
        if (inputFilename == self.inputFilename) or (inputFilename == ""):
            pass
        else:
            self.inputFilename = inputFilename
            self.inputFilenameSetText = self.inputFilename # os.path.basename(self.inputFilename)
            self.lineEditInputFilename.setText(self.inputFilenameSetText)

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
                self.bestRedshiftErr = None
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
        if not isinstance(self.inputFilename, (list, np.ndarray)) and not hasattr(self.inputFilename, 'read') and not os.path.isfile(self.inputFilename) and self.inputFilename.split('-')[0] not in list(catalogDict.keys()):   # Not an array and not a file-handle and not a file and not a catalog input
            QtGui.QMessageBox.critical(self, "Error", "File not found!")
            return
        try:
            if self.inputFilename.split('-')[0] in list(catalogDict.keys()):
                knownZ = 0
                self.bestRedshift = 0
                self.bestRedshiftErr = None
                self.lineEditKnownZ.setText("")
        except Exception as e:
            pass  # Not an Open Supernova catalog object. Make this neater in future.

        if self.checkBoxRlap.isChecked():
            self.getRlapScores = True
        else:
            self.getRlapScores = False

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
        self.bestTypes, self.softmax, self.idx, self.typeNamesList, self.inputImageUnRedshifted, self.inputMinMaxIndex = spectrumInfo
        self.progressBar.setValue(85)#self.progressBar.value()+)
        self.done_fit_thread_single_redshift()

    def done_fit_thread_single_redshift(self):
        if not self.cancelledFitting:
            self.plotted = True
            self.list_best_matches_single_redshift()
            self.set_plot_redshift(self.bestRedshift)
            self.plot_cross_corr()
            self.progressBar.setValue(100)
            # QtGui.QMessageBox.information(self, "Done!", "Finished Fitting Input Spectrum")

    def best_broad_type(self):
        bestMatchList = []
        for i in range(10):
            host, name, age = classification_split(self.bestTypes[i])
            bestMatchList.append([host, name, age, self.softmax[i]])
        host, prevName, bestAge, probTotal, reliableFlag = combined_prob(bestMatchList)
        self.labelBestSnType.setText(prevName)
        self.labelBestAgeRange.setText(bestAge)
        self.labelBestHostType.setText(host)
        if host == "":
            self.labelBestHostType.setFixedWidth(0)
        if self.bestRedshiftErr is None:
            self.labelBestRedshift.setText(str(self.bestRedshift))
        else:
            self.labelBestRedshift.setText("{} {} {}".format(str(self.bestRedshift), "±", self.bestRedshiftErr))
        self.labelBestRelProb.setText("%s%%" % str(round(100*probTotal, 2)))
        if host == "":                                     
            self.labelBestHostType.setFixedWidth(0)
        if reliableFlag:
            self.labelInconsistentWarning.setText("Reliable matches")
            self.labelInconsistentWarning.setStyleSheet('color: green')
        else:
            self.labelInconsistentWarning.setText("Unreliable matches")
            self.labelInconsistentWarning.setStyleSheet('color: red')

    def get_smoothed_templates(self, snName, snAge, hostName):
        snInfos, snNames, hostInfos, hostNames = self.get_sn_and_host_templates(snName, snAge, hostName)
        fluxes = []
        minMaxIndexes = []
        for i in range(len(snNames)):
            wave, flux, minMaxIndex = combined_sn_and_host_data(snCoeff=1, galCoeff=0, z=0, snInfo=snInfos[i], galInfo=hostInfos[0], w0=self.w0, w1=self.w1, nw=self.nw)

            fluxes.append(flux)
            minMaxIndexes.append(minMaxIndex)

        return fluxes, snNames, minMaxIndexes

    def low_rlap_warning_label(self, bestName, bestAge, bestHost):
        fluxes, snNames, templateMinMaxIndexes = self.get_smoothed_templates(bestName, bestAge, bestHost)
        rlapCalc = RlapCalc(self.inputImageUnRedshifted, fluxes, snNames, self.wave, self.inputMinMaxIndex, templateMinMaxIndexes)
        rlapLabel, rlapWarning = rlapCalc.rlap_label()

        return rlapLabel, rlapWarning

    def list_best_matches_single_redshift(self):
        print("listing best matches...")
        redshifts, redshiftErrs = self.best_redshifts()
        self.listWidget.clear()

        header = ['No.', 'Type', 'Age', 'Softmax Prob.']
        if self.classifyHost:
            header.insert(1, 'Host')
        if not self.knownRedshift:
            header.insert(3, 'Redshift')
        if self.getRlapScores:
            header.insert(5, 'rlap')
        self.listWidget.addItem("".join(word.ljust(25) for word in header))

        for i in range(20):
            host, name, age = classification_split(self.bestTypes[i])
            prob = self.softmax[i]
            redshift = redshifts[i]

            line = [str(i + 1), name, age, str(prob)]
            if self.classifyHost:
                line.insert(1, host)
            if not self.knownRedshift:
                line.insert(3, str(redshift))
            if self.getRlapScores:
                fluxes, snNames, templateMinMaxIndexes = self.get_smoothed_templates(name, age, host)
                rlapCalc = RlapCalc(self.inputImageUnRedshifted, fluxes, snNames, self.wave, self.inputMinMaxIndex, templateMinMaxIndexes)
                rlap = rlapCalc.rlap_label()[0]
                line.insert(5, str(rlap))
            self.listWidget.addItem("".join(word.ljust(25) for word in line))

            if i == 0:
                SNTypeComboBoxIndex = self.comboBoxSNType.findText(name)
                self.comboBoxSNType.setCurrentIndex(SNTypeComboBoxIndex)
                AgeComboBoxIndex = self.comboBoxAge.findText(age)
                self.comboBoxAge.setCurrentIndex(AgeComboBoxIndex)
                hostComboBoxIndex = self.comboBoxHost.findText(host)
                self.comboBoxHost.setCurrentIndex(hostComboBoxIndex)

                rlap, rlapWarning = self.low_rlap_warning_label(name, age, host)
                if rlapWarning:
                    self.labelRlapWarning.setText("Low rlap: {0}".format(rlap))
                    self.labelRlapWarning.setStyleSheet('color: red')
                else:
                    self.labelRlapWarning.setText("Good rlap: {0}".format(rlap))
                    self.labelRlapWarning.setStyleSheet('color: green')

            if not self.knownRedshift:
                self.bestRedshift = redshifts[0]
                self.bestRedshiftErr = redshiftErrs[0]
        self.best_broad_type()

    def list_item_clicked(self, item):
        if item.text()[0].isdigit():
            host = "No Host"
            if self.knownRedshift:
                if self.classifyHost:
                    if self.getRlapScores:
                        index, host, snTypePlot, age1, to, age3, softmax, rlap = str(item.text()).split()
                    else:
                        index, host, snTypePlot, age1, to, age3, softmax = str(item.text()).split()
                else:
                    if self.getRlapScores:
                        index, snTypePlot, age1, to, age3, softmax, rlap = str(item.text()).split()
                    else:
                        index, snTypePlot, age1, to, age3, softmax = str(item.text()).split()
            else:
                if self.classifyHost:
                    if self.getRlapScores:
                        index, host, snTypePlot, age1, to, age3, redshift, softmax, rlap = str(item.text()).split()
                    else:
                        index, host, snTypePlot, age1, to, age3, redshift, softmax = str(item.text()).split()
                else:
                    if self.getRlapScores:
                        index, snTypePlot, age1, to, age3, redshift, softmax, rlap = str(item.text()).split()
                    else:
                        index, snTypePlot, age1, to, age3, redshift, softmax = str(item.text()).split()
                self.set_plot_redshift(redshift)
            agePlot = age1 + ' to ' + age3

            self.plot_cross_corr()
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
            self.graphicsView.setXRange(int(self.w0), int(self.w1))
            self.graphicsView.setYRange(0, 1)
            self.graphicsView.plotItem.showGrid(x=True, y=True, alpha=0.95)
            self.graphicsView.plotItem.setLabels(bottom="Observed Wavelength (<font>&#8491;</font>)")

            try:
                self.cAxis.setScale(1/(1+self.plotZ))
            except ZeroDivisionError:
                print("Invalid redshift. Redshift cannot be -1.")
            self.cAxis.setGrid(False)
            self.cAxis.setLabel("Rest Wavelength (<font>&#8491;</font>)")

            if np.any(self.templatePlotFlux):
                rlapCalc = RlapCalc(self.inputImageUnRedshifted, [self.templatePlotFlux], [self.templatePlotName], self.wave, self.inputMinMaxIndex, [self.templateMinMaxIndex])
                rlap = rlapCalc.rlap_label()[0]
                self.labelRlapScore.setText("rlap: {0}".format(rlap))

    def best_redshifts(self):
        redshifts = []
        redshiftErrs = []
        for i in range(20):
            host, name, age = classification_split(self.bestTypes[i])
            redshift, crossCorr, medianName, redshiftErr = self.calc_redshift(name, age)
            redshifts.append(redshift)
            redshiftErrs.append(redshiftErr)
        return redshifts, redshiftErrs

    def set_template_sub_index(self, templateName):
        snInfos, snNames, hostInfos, hostNames = self.get_sn_and_host_templates(self.snName, self.snAge, self.hostName)
        if snNames.size:
            self.templateSubIndex = np.where(snNames == templateName)[0][0]

    def calc_redshift(self, snName, snAge):
        host = "No Host"
        snInfos, snNames, hostInfos, hostNames = self.get_sn_and_host_templates(snName, snAge, host)
        numOfSubTemplates = len(snNames)
        templateNames = snNames
        templateFluxes = []
        templateMinMaxIndexes = []
        for i in range(numOfSubTemplates):
            templateFluxes.append(snInfos[i][1])
            templateMinMaxIndexes.append((snInfos[i][2], snInfos[i][3]))

        redshift, crossCorrs, medianName, redshiftErr = get_median_redshift(self.inputImageUnRedshifted, templateFluxes, self.nw, self.dwlog, self.inputMinMaxIndex, templateMinMaxIndexes, templateNames, outerVal=0.5)
        if redshift is None:
            return 0, 0, "", 0

        return round(redshift, 4), crossCorrs, medianName, round(redshiftErr, 4)

    def plot_cross_corr(self):
        zAxis = get_redshift_axis(self.nw, self.dwlog)
        if type(self.crossCorrs) == dict:
            crossCorr = np.real(self.crossCorrs["_".join(self.templatePlotName.split('_')[:-1])])
        else:
            crossCorr = np.zeros(self.nw)
        self.graphicsView_2.clear()
        self.graphicsView_2.plot(zAxis, crossCorr)
        self.graphicsView_2.setXRange(0, 1)
        self.graphicsView_2.setYRange(min(crossCorr), max(crossCorr))
        self.graphicsView_2.plotItem.showGrid(x=True, y=True, alpha=0.95)
        # self.graphicsView_2.plotItem.setLabels(bottom="z")
        self.graphicsView_2.addItem(self.infLine)

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
        self.inputFilename = inputFilename
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
        loadInputSpectraUnRedshifted = LoadInputSpectra(self.inputFilename, 0, self.smooth, trainParams, self.minWave, self.maxWave, self.classifyHost)
        inputImageUnRedshifted, inputRedshift, typeNamesList, nw, nBins, minMaxIndexUnRedshifted = loadInputSpectraUnRedshifted.input_spectra()

        loadInputSpectra = LoadInputSpectra(self.inputFilename, self.knownZ, self.smooth, trainParams, self.minWave, self.maxWave, self.classifyHost)
        inputImage, inputRedshift, typeNamesList, nw, nBins, minMaxIndex = loadInputSpectra.input_spectra()
        bestTypesList = BestTypesListSingleRedshift(self.modelFilename, inputImage, typeNamesList, nw, nBins)
        bestTypes = bestTypesList.bestTypes[0]
        softmax = bestTypesList.softmaxOrdered[0]
        idx = bestTypesList.idx[0]

        return bestTypes, softmax, idx, typeNamesList, inputImageUnRedshifted[0], minMaxIndexUnRedshifted[0]

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
