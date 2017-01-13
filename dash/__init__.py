import sys
import os
import pickle
mainDirectory = os.path.dirname(os.path.abspath(__file__))
#from main import *
from PyQt4 import QtGui

from classify import Classify


def main():
    classification = Classify(filenames=['test_spectrum_file.dat',
                                         'test_spectrum_file.dat'],
                              redshifts=[0.34, 0.13])
    print classification.list_best_matches()
    # classification.plot_with_gui(indexToPlot=1)


def run_gui():
    app = QtGui.QApplication(sys.argv)
    form = MainApp()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
    # run_gui()


