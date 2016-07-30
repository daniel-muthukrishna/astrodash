from PyQt4 import QtGui
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
        
        self.loadInputSpectra = LoadInputSpectra('Ia/sn1981b.max.dat', self.minZ, self.maxZ)
        self.inputImages, self.inputLabels, self.inputRedshifts = self.loadInputSpectra.input_spectra()
        self.bestTypesList = BestTypesList("/tmp/model.ckpt", self.inputImages, self.inputLabels, self.inputRedshifts)
        self.bestForEachType, self.typeNamesList, self.redshiftIndex = self.bestTypesList.print_list()
        self.templateFluxes, self.inputFluxes = self.bestTypesList.plot_best_types()
        self.inputRedshifts, self.redshiftGraphs = self.bestTypesList.redshift_graph()
                
        self.list_best_matches()
        self.plot_best_matches(0)
        self.plot_redshift_graphs(0)

    def list_best_matches(self):
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
        self.graphicsView.clear()
        self.graphicsView.addLegend()
        #templateFluxes, inputFluxes = self.bestTypesList.plot_best_types()
        self.graphicsView.plot(self.inputFluxes[indexToPlot], name='Input Spectra', pen={'color': (0,255,0)})
        self.graphicsView.plot(self.templateFluxes[indexToPlot], name='Template', pen={'color': (255,0,0)})

    def plot_redshift_graphs(self, indexToPlot):
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

    










def main():
    app = QtGui.QApplication(sys.argv)
    form = MainApp()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()
