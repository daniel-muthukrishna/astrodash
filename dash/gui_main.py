import os
import sys
import pickle

from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal

from dash.design import Ui_MainWindow

mainDirectory = os.path.dirname(os.path.abspath(__file__))

from dash.restore_model import *
from dash.create_arrays import AgeBinning

class MainApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None, inputFilename="DefaultFilename"):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)

        self.templates()
        self.plotted = False
        self.indexToPlot = 0
        self.plotZ = 0
        self.templateFluxes = np.zeros((2, int(self.nw)))
        self.inputFluxes = np.zeros((2, int(self.nw)))
        self.inputImageUnRedshifted = np.zeros((2, int(self.nw)))
        self.templatePlotFlux = np.zeros(int(self.nw))
        self.templateSubIndex = 0

        self.mainDirectory = os.path.dirname(os.path.abspath(__file__))

        self.pushButtonLeftTemplate.clicked.connect(self.select_sub_template_left)
        self.pushButtonRightTemplate.clicked.connect(self.select_sub_template_right)
        self.btnBrowse.clicked.connect(self.select_input_file)
        self.listWidget.itemClicked.connect(self.list_item_clicked)
        self.btnRefit.clicked.connect(self.fit_spectra)
        self.inputFilename = inputFilename
        self.progressBar.setValue(100)
        self.add_combo_box_entries()

        self.checkBoxZeroZTrained.stateChanged.connect(self.zero_redshift_model)
        self.checkBoxKnownZ.stateChanged.connect(self.zero_redshift_model)
        self.checkBoxAgnosticZTrained.stateChanged.connect(self.agnostic_redshift_model)

        self.checkBoxZeroZTrained.setChecked(True)
        self.checkBoxKnownZ.setChecked(True)
        self.checkBoxAgnosticZTrained.setEnabled(False)
        self.checkBoxGalTrained.setEnabled(False)
        self.comboBoxHost.setEnabled(False)

        self.horizontalSliderSmooth.valueChanged.connect(self.smooth_slider_changed)
        self.lineEditSmooth.textChanged.connect(self.smooth_text_changed)

        self.horizontalSliderRedshift.valueChanged.connect(self.redshift_slider_changed)
        self.lineEditRedshift.textChanged.connect(self.redshift_text_changed)

        self.comboBoxSNType.currentIndexChanged.connect(self.combo_box_changed)
        self.comboBoxAge.currentIndexChanged.connect(self.combo_box_changed)

    def templates(self):
        with open(os.path.join(mainDirectory, "training_params_v02.pickle"), 'rb') as f:
            pars = pickle.load(f)
        self.nTypes = pars['nTypes']
        self.nw = pars['nw']
        w0, w1, minAge, maxAge, ageBinSize, self.typeList = pars['w0'], pars['w1'], pars['minAge'], pars['maxAge'], \
                                                       pars['ageBinSize'], pars['typeList']

        dwlog = np.log(w1/w0)/self.nw
        self.wave = w0 * np.exp(np.arange(0,self.nw) * dwlog)

        loaded = np.load(os.path.join(mainDirectory, 'templates_v02.npz'))
        self.templateFluxesAll = loaded['templateFluxesAll']
        self.templateFileNamesAll = loaded['templateFilenamesAll']

    def select_sub_template_right(self):
        self.templateSubIndex += 1
        self.plot_sub_templates()

    def select_sub_template_left(self):
        self.templateSubIndex -= 1
        self.plot_sub_templates()


    def plot_sub_templates(self):
        numOfSubTemplates = len(self.templateFileNamesAll[self.templateIndex])
        if self.templateSubIndex >= numOfSubTemplates:
            self.templateSubIndex = 0
        if self.templateSubIndex < 0:
            self.templateSubIndex = numOfSubTemplates - 1

        self.templatePlotFlux = self.templateFluxesAll[self.templateIndex][self.templateSubIndex]
        self.templatePlotName = self.templateFileNamesAll[self.templateIndex][self.templateSubIndex]
        print(self.templatePlotName)
        self.plot_best_matches()

    def combo_box_changed(self):
        self.templateIndex = self.comboBoxSNType.currentIndex() * (self.nTypes + 1) +  self.comboBoxAge.currentIndex()

        self.templatePlotFlux = self.templateFluxesAll[self.templateIndex][0]
        self.templatePlotName = self.templateFileNamesAll[self.templateIndex][0]
        print(self.templatePlotName)
        self.plot_best_matches()



    def add_combo_box_entries(self):
        minAge = -20
        maxAge = 50
        ageBinSize = 4
        ageLabels = AgeBinning(minAge, maxAge, ageBinSize).age_labels()
        for i in range(len(ageLabels)):
            self.comboBoxAge.addItem(ageLabels[i])

        for typeName in self.typeList:
            self.comboBoxSNType.addItem(typeName)


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

    def zero_redshift_model(self):
        if (self.checkBoxZeroZTrained.isChecked() == True):
            self.modelFilename = os.path.join(self.mainDirectory, "model_trainedAtZeroZ.ckpt")
            self.checkBoxKnownZ.setEnabled(True)
            self.lineEditKnownZ.setEnabled(True)
            self.checkBoxAgnosticZTrained.setEnabled(False)
            self.checkBoxAgnosticZTrained.setChecked(False)

            if (self.checkBoxKnownZ.isChecked() == True):
                self.lineEditMinZ.setEnabled(False)
                self.lineEditMaxZ.setEnabled(False)
            else:
                self.lineEditMinZ.setEnabled(True)
                self.lineEditMaxZ.setEnabled(True)

        elif (self.checkBoxZeroZTrained.isChecked() == False):
            self.checkBoxKnownZ.setEnabled(False)
            self.lineEditKnownZ.setEnabled(False)
            self.checkBoxZeroZTrained.setEnabled(False)
            self.checkBoxZeroZTrained.setChecked(False)
            self.checkBoxAgnosticZTrained.setEnabled(True)
            self.checkBoxAgnosticZTrained.setChecked(True)
            self.agnostic_redshift_model()

    def agnostic_redshift_model(self):
        if (self.checkBoxAgnosticZTrained.isChecked() == True):
            self.checkBoxZeroZTrained.setEnabled(False)
            self.checkBoxZeroZTrained.setChecked(False)
            self.modelFilename = os.path.join(self.mainDirectory, "model_agnostic_redshift.ckpt")
        elif (self.checkBoxAgnosticZTrained.isChecked() == False):
            self.checkBoxZeroZTrained.setEnabled(True)
            self.checkBoxZeroZTrained.setChecked(True)
            self.zero_redshift_model()




        
    def select_input_file(self):
        inputFilename = QtGui.QFileDialog.getOpenFileName(self,"Select a spectrum file")[0]
        print(inputFilename)
        print(self.inputFilename)
        if (inputFilename == self.inputFilename) or (inputFilename == ""):
            pass
        else:
            self.inputFilename = inputFilename
            self.lblInputFilename.setText(inputFilename.split('/')[-1])

            #Run InputSpectra

            self.fit_spectra()

    def fit_spectra(self):
        self.cancelledFitting = False
        self.progressBar.setMaximum(100) #
        self.progressBar.setValue(36)
        try:
            self.smooth = int(self.lineEditSmooth.text())
        except ValueError:
            QtGui.QMessageBox.critical(self, "Error", "Smooth must be positive integer")

        if (self.checkBoxKnownZ.isChecked() == True):
            self.redshiftFlag = True
            try:
                self.knownZ = float(self.lineEditKnownZ.text())
            except ValueError:
                QtGui.QMessageBox.critical(self, "Error", "Enter Known Redshift")
            self.minZ = self.knownZ
            self.maxZ = self.knownZ
            self.set_plot_redshift(self.knownZ)

            self.fitThread = FitSpectrumThread(self.inputFilename, self.minZ, self.maxZ, self.redshiftFlag, self.modelFilename, self.smooth)
            self.fitThread.trigger.connect(self.load_spectrum_single_redshift)


        else:
            self.redshiftFlag = False
            self.minZ = float(self.lineEditMinZ.text())
            self.maxZ = float(self.lineEditMaxZ.text())
            self.knownZ = self.minZ
            print(self.minZ, self.maxZ)
            self.fitThread = FitSpectrumThread(self.inputFilename, self.minZ, self.maxZ, self.redshiftFlag, self.modelFilename, self.smooth)
            self.connect(self.fitThread, SIGNAL("load_spectrum(PyQt_PyObject)"), self.load_spectrum)
            self.connect(self.fitThread, SIGNAL("finished()"), self.done_fit_thread)

        self.fitThread.start()

        self.btnCancel.clicked.connect(self.cancel)


    def cancel(self):
        if (self.cancelledFitting == False):
            self.cancelledFitting = True
            self.fitThread.terminate()
            self.progressBar.setValue(100)
            QtGui.QMessageBox.information(self, "Cancelled!", "Stopped Fitting Input Spectrum")


    def load_spectrum_single_redshift(self, spectrumInfo):
        self.bestTypes, self.softmax, self.idx, self.templateFluxes, self.inputFluxes, self.typeNamesList, self.inputImageUnRedshifted = spectrumInfo
        self.progressBar.setValue(85)#self.progressBar.value()+)
        self.done_fit_thread_single_redshift()

    def done_fit_thread_single_redshift(self):
        if (self.cancelledFitting == False):
            self.plotted = True
            self.list_best_matches_single_redshift()
            self.plot_best_matches()
            self.progressBar.setValue(100)
            QtGui.QMessageBox.information(self, "Done!", "Finished Fitting Input Spectrum")

    def list_best_matches_single_redshift(self):
        print("listing best matches...")
        self.listWidget.clear()
        self.listWidget.addItem("".join(word.ljust(25) for word in ['No.', 'Type', 'Age', 'Softmax Prob..']))
        for i in range(20):
            name, age = self.bestTypes[i].split(': ')
            self.listWidget.addItem("".join(word.ljust(25) for word in [str(i+1), name, age, str(self.softmax[i])]))

            if i == 0:
                SNTypeComboBoxIndex = self.comboBoxSNType.findText(name)
                self.comboBoxSNType.setCurrentIndex(SNTypeComboBoxIndex)
                AgeComboBoxIndex = self.comboBoxAge.findText(age)
                self.comboBoxAge.setCurrentIndex(AgeComboBoxIndex)

    def load_spectrum(self, spectrumInfo):
        self.bestForEachType, self.templateFluxes, self.inputFluxes, self.inputRedshifts, self.redshiftGraphs, self.typeNamesList = spectrumInfo
        self.progressBar.setValue(85)#self.progressBar.value()+)

    def done_fit_thread(self):
        if (self.cancelledFitting == False):
            self.list_best_matches()
            self.plot_best_matches()
            self.plot_redshift_graphs()
            self.progressBar.setValue(100)
            QtGui.QMessageBox.information(self, "Done!", "Finished Fitting Input Spectrum")
        

    def list_best_matches(self):
        print("listing best matches...")
        self.listWidget.clear()
        self.listWidget.addItem("".join(word.ljust(25) for word in ['No.', 'Type', 'Age', 'Redshift', 'Rel. Prob.']))
        for i in range(20): #len(bestForEachType)
            bestIndex = int(self.bestForEachType[i][0])
            name, age = self.typeNamesList[bestIndex].split(': ')
            self.listWidget.addItem("".join(word.ljust(25) for word in [str(i+1), name, age , str(self.bestForEachType[i][1]), str(self.bestForEachType[i][2])]))


    def list_item_clicked(self, item):
        index, self.SNTypePlot, age1, age2, age3, softmax = str(item.text()).split()
        self.AgePlot = age1 + ' to ' + age3

        try:
            self.indexToPlot = int(index) - 1 #Two digit numbers
        except ValueError:
            self.indexToPlot = 0

        SNTypeComboBoxIndex = self.comboBoxSNType.findText(self.SNTypePlot)
        self.comboBoxSNType.setCurrentIndex(SNTypeComboBoxIndex)
        AgeComboBoxIndex = self.comboBoxAge.findText(self.AgePlot)
        self.comboBoxAge.setCurrentIndex(AgeComboBoxIndex)

        if self.redshiftFlag == False:
            self.plot_redshift_graphs()

        
    def plot_best_matches(self):
        if self.plotted == True:
            templateWave = self.wave * (1 + (self.plotZ))
            self.labelTemplateName.setText(self.templatePlotName)

            self.graphicsView.clear()

            if self.redshiftFlag == True:
                inputPlotFlux = self.inputImageUnRedshifted[0]
            elif self.redshiftFlag == False:
                inputPlotFlux = self.inputFluxes[self.indexToPlot]

            self.graphicsView.plot(self.wave, inputPlotFlux, name='Input Spectrum', pen={'color': (0, 255, 0)})
            self.graphicsView.plot(templateWave, self.templatePlotFlux, name=self.templatePlotName, pen={'color': (255,0,0)})
            self.graphicsView.setRange(xRange=[2500,10000])




    def plot_redshift_graphs(self):
        print("listing Redshift Graphs...")
        print(len(self.inputRedshifts), len(self.redshiftGraphs[self.indexToPlot]))
        self.graphicsView_2.clear()
        self.graphicsView_2.plot(self.inputRedshifts, self.redshiftGraphs[self.indexToPlot])
        self.graphicsView_2.setLabels(left=("Rel. Prob."), bottom=("z"))
        
    def browse_folder(self):
        self.listWidget.clear()
        directory = QtGui.QFileDialog.getExistingDirectory(self,"Pick a folder")

        if directory:
            for file_name in os.listdir(directory):
                self.listWidget.addItem(file_name)

    
class FitSpectrumThread(QThread):

    trigger = pyqtSignal('PyQt_PyObject')

    def __init__(self, inputFilename, minZ, maxZ, redshiftFlag, modelFilename, smooth):
        QThread.__init__(self)
        self.inputFilename = str(inputFilename)
        self.minZ = minZ
        self.maxZ = maxZ
        self.redshiftFlag = redshiftFlag
        self.modelFilename = modelFilename
        self.smooth = smooth

    def __del__(self):
        self.wait()

    def _input_spectrum(self):
        trainParams = get_training_parameters()
        loadInputSpectra = LoadInputSpectra(self.inputFilename, self.minZ, self.maxZ, self.smooth, trainParams)
        inputImages, inputRedshifts, typeNamesList, nw, nTypes = loadInputSpectra.input_spectra()
        bestTypesList = BestTypesList(self.modelFilename, inputImages, inputRedshifts, typeNamesList, nw, nTypes)
        bestForEachType, redshiftIndex = bestTypesList.print_list()
        templateFluxes, inputFluxes = bestTypesList.plot_best_types()
        inputRedshifts, redshiftGraphs = bestTypesList.redshift_graph()

        return (bestForEachType, templateFluxes, inputFluxes,
                inputRedshifts, redshiftGraphs, typeNamesList)

    def _input_spectrum_single_redshift(self):
        trainParams = get_training_parameters()
        loadInputSpectraUnRedshifted = LoadInputSpectra(self.inputFilename, 0, 0, self.smooth, trainParams)
        inputImageUnRedshifted, inputRedshift, typeNamesList, nw, nBins = loadInputSpectraUnRedshifted.input_spectra()

        trainParams = get_training_parameters()
        loadInputSpectra = LoadInputSpectra(self.inputFilename, self.minZ, self.maxZ, self.smooth, trainParams)
        inputImage, inputRedshift, typeNamesList, nw, nBins = loadInputSpectra.input_spectra()
        bestTypesList = BestTypesListSingleRedshift(self.modelFilename, inputImage, typeNamesList, nw, nBins)
        bestTypes = bestTypesList.bestTypes[0]
        softmax = bestTypesList.softmaxOrdered[0]
        idx = bestTypesList.idx[0]
        templateFluxes, inputFluxes = bestTypesList.plot_best_types()

        return bestTypes, softmax, idx, templateFluxes, inputFluxes, typeNamesList, inputImageUnRedshifted

    def run(self):
        if self.redshiftFlag == True:
            spectrumInfo = self._input_spectrum_single_redshift()
            self.trigger.emit(spectrumInfo)
        else:
            spectrumInfo = self._input_spectrum()
            self.emit(SIGNAL('load_spectrum(PyQt_PyObject)'), spectrumInfo)
        







def main():
    app = QtGui.QApplication(sys.argv)
    form = MainApp()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()
