from PyQt4 import QtGui
import sys
import os
import pickle

import design

class MainApp(QtGui.QMainWindow, design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)

        self.inputFilename = self.btnBrowse.clicked.connect(self.select_input_file)
        ######INPUT_SPECTRA_DETAILS######
        with open('../training_params.pickle') as f:
            self.nTypes, self.w0, self.w1, self.nw, self.minAge, self.maxAge, self.ageBinSize = pickle.load(f)
        

    def select_input_file(self):
        inputFilename = QtGui.QFileDialog.getOpenFileName(self,"Select a spectrum file")
        self.lblInputFilename.setText(inputFilename.split('/')[-1])

        return inputFilename

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
