import sys
import os
mainDirectory = os.path.dirname(os.path.abspath(__file__))

# try:
#     from PyQt4 import QtGui
# except ImportError:
#     from main import *
#     print "Warning: You will need to install 'PyQt4' if you want to use the graphical interface. " \
#           "Using the automatic library will continue to work."

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


