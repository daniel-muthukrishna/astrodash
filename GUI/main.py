from PyQt4 import QtGui
from PyQt4.QtCore import QThread, SIGNAL
import sys
import os
import pickle

import design
import sys
sys.path.insert(0, '../')

from restore_model import *

class MainApp(QtGui.QMainWindow, design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)

        self.btnBrowse.clicked.connect(self.select_input_file)
        self.listWidget.itemClicked.connect(self.list_item_clicked)
        self.btnRefit.clicked.connect(self.fit_spectra)
        self.inputFilename = "DefaultFilename"
        self.progressBar.setValue(100)

        
    def select_input_file(self):
        inputFilename = QtGui.QFileDialog.getOpenFileName(self,"Select a spectrum file")
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
        self.minZ = float(self.lineEditMinZ.text())
        self.maxZ = float(self.lineEditMaxZ.text())
        print (self.minZ, self.maxZ)
        self.cancelledFitting = False

        self.progressBar.setMaximum(100) #
        self.progressBar.setValue(36)
        self.fitThread = FitSpectrumThread(self.inputFilename, self.minZ, self.maxZ)
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


    def load_spectrum(self, spectrumInfo):
        self.bestForEachType, self.templateFluxes, self.inputFluxes, self.inputRedshifts, self.redshiftGraphs = spectrumInfo
        self.progressBar.setValue(85)#self.progressBar.value()+)


    def done_fit_thread(self):
        if (self.cancelledFitting == False):
            self.list_best_matches()
            self.plot_best_matches(0)
            self.plot_redshift_graphs(0)
            self.progressBar.setValue(100)
            QtGui.QMessageBox.information(self, "Done!", "Finished Fitting Input Spectrum")
        

    def list_best_matches(self):
        print("listing best matches...")
        self.listWidget.clear()
        self.listWidget.addItem("".join(word.ljust(25) for word in ['No.', 'Type', 'Age', 'Redshift', 'Rel. Prob.']))
        for i in range(20): #len(bestForEachType)
            bestIndex = int(self.bestForEachType[i][0])
            name, age = typeNamesList[bestIndex].split(': ')
            self.listWidget.addItem("".join(word.ljust(25) for word in [str(i+1), name, age , str(self.bestForEachType[i][1]), str(self.bestForEachType[i][2])]))

    def list_item_clicked(self, item):
        try:
            indexToPlot = int(item.text()[0]) - 1
        except ValueError:
            indexToPlot = 0
        self.plot_best_matches(indexToPlot)
        self.plot_redshift_graphs(indexToPlot)
        
    def plot_best_matches(self, indexToPlot):
        print("plotting best matches...")
        self.graphicsView.clear()
        self.graphicsView.addLegend()
        #templateFluxes, inputFluxes = self.bestTypesList.plot_best_types()
        self.graphicsView.plot(self.inputFluxes[indexToPlot], name='Input Spectra', pen={'color': (0,255,0)})
        self.graphicsView.plot(self.templateFluxes[indexToPlot], name='Template', pen={'color': (255,0,0)})

    def plot_redshift_graphs(self, indexToPlot):
        print("listing Redshift Graphs...")
        print(len(self.inputRedshifts), len(self.redshiftGraphs[indexToPlot]))
        self.graphicsView_2.clear()
        self.graphicsView_2.plot(self.inputRedshifts, self.redshiftGraphs[indexToPlot])
        self.graphicsView_2.setLabels(left=("Rel. Prob."), bottom=("z"))
        
    def browse_folder(self):
        self.listWidget.clear()
        directory = QtGui.QFileDialog.getExistingDirectory(self,"Pick a folder")

        if directory:
            for file_name in os.listdir(directory):
                self.listWidget.addItem(file_name)

    
class FitSpectrumThread(QThread):
    def __init__(self, inputFilename, minZ, maxZ):
        QThread.__init__(self)
        self.inputFilename = inputFilename
        self.minZ = minZ
        self.maxZ = maxZ

    def __del__(self):
        self.wait()

    def _input_spectrum(self):
        loadInputSpectra = LoadInputSpectra('Ia/sn1981b.max.dat', self.minZ, self.maxZ)
        inputImages, inputLabels, inputRedshifts = loadInputSpectra.input_spectra()
        bestTypesList = BestTypesList("/tmp/model.ckpt", inputImages, inputLabels, inputRedshifts)
        bestForEachType, typeNamesList, redshiftIndex = bestTypesList.print_list()
        templateFluxes, inputFluxes = bestTypesList.plot_best_types()
        inputRedshifts, redshiftGraphs = bestTypesList.redshift_graph()

        return (bestForEachType, templateFluxes, inputFluxes,
                inputRedshifts, redshiftGraphs)

    def run(self):
         spectrumInfo = self._input_spectrum()
         self.emit(SIGNAL('load_spectrum(PyQt_PyObject)'), spectrumInfo)
        







def main():
    app = QtGui.QApplication(sys.argv)
    form = MainApp()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()
