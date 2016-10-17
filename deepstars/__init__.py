import sys
import os
mainDirectory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(mainDirectory, "../GUI"))

from main import *
from PyQt4 import QtGui




def main():
    app = QtGui.QApplication(sys.argv)
    form = MainApp()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()