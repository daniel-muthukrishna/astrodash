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

        self.inputFilename = self.btnBrowse.clicked.connect(self.select_input_file)
        self.typeIndex = self.listWidget.itemClicked.connect(self.list_item_clicked)
        ######INPUT_SPECTRA_DETAILS######
        with open('../training_params.pickle') as f:
            self.nTypes, self.w0, self.w1, self.nw, self.minAge, self.maxAge, self.ageBinSize = pickle.load(f)

        self.bestTypesList = BestTypesList("/tmp/model.ckpt")

        self.templateFluxes, self.inputFluxes = self.bestTypesList.plot_best_types()
        

    def select_input_file(self):
        inputFilename = QtGui.QFileDialog.getOpenFileName(self,"Select a spectrum file")
        self.lblInputFilename.setText(inputFilename.split('/')[-1])

        #Run InputSpectra

        self.list_best_matches()
        self.plot_best_matches(0)

        return inputFilename

    def list_best_matches(self):
        self.listWidget.clear()
        bestForEachType, typeNamesList, redshiftIndex = self.bestTypesList.create_list()
        self.listWidget.addItem("".join(word.ljust(25) for word in ['No.', 'Type', 'Age', 'Redshift', 'Rel. Prob.']))
        for i in range(20): #len(bestForEachType)
            bestIndex = int(bestForEachType[i][0])
            name, age = typeNamesList[bestIndex].split(': ')
            self.listWidget.addItem("".join(word.ljust(25) for word in [str(i+1), name, age , str(bestForEachType[i][1]), str(bestForEachType[i][2])]))

    def list_item_clicked(self, item):
        try:
            indexToPlot = int(item.text()[0]) - 1
        except ValueError:
            indexToPlot = 0
        self.plot_best_matches(indexToPlot)
        
    def plot_best_matches(self, indexToPlot):
        self.graphicsView.clear()
        self.graphicsView.addLegend()
        #templateFluxes, inputFluxes = self.bestTypesList.plot_best_types()
        self.graphicsView.plot(self.inputFluxes[indexToPlot], name='Input Spectra', pen={'color': (0,255,0)})
        self.graphicsView.plot(self.templateFluxes[indexToPlot], name='Template', pen={'color': (255,0,0)})

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
