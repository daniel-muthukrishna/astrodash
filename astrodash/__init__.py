import sys
import argparse
from astrodash.download_data_files import download_all_files

download_all_files('v06')

try:
    from PyQt5 import QtGui
    from astrodash.gui_main import MainApp
except ImportError:
    print("Warning: You will need to install 'PyQt5' if you want to use the graphical interface. " \
          "Using the automatic library will continue to work.")

from astrodash.classify import Classify


def main():
    classification = Classify(filenames=['test_spectrum_file.dat',
                                         'test_spectrum_file.dat'],
                              redshifts=[0.34, 0.13])
    print(classification.list_best_matches())
    # classification.plot_with_gui(indexToPlot=1)


def run_gui():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str, help='Optional: Spectrum filepath', action='store_true')
    parser.add_argument('redshift', type=float, help='Optional: Spectrum redshift', action='store_true')
    parser.add_argument('--classify_host', type=bool, help='Optional: Classify Host', action='store_true')
    parser.add_argument('smooth', type=bool, help='Optional: Smooth spectrum', action='store_true')
    args = parser.parse_args()

    app = QtGui.QApplication(sys.argv)

    if args.filepath is None:
        filepath = "DefaultFilename"
    else:
        filepath = args.filepath
    form = MainApp(inputFilename=filepath)

    if args.redshift is None:
        knownZ = False
        redshift = ""
    else:
        knownZ = True
        redshift = str(args.redshift)
    form.checkBoxKnownZ.setChecked(knownZ)
    form.lineEditKnownZ.setText(str(redshift))

    if args.classify_host:
        form.checkBoxClassifyHost.setChecked(True)
        form.classifyHost = True

    if args.smooth is not None:
        form.lineEditSmooth.setText(str(args.smooth))

    form.select_tensorflow_model()
    form.fit_spectra()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
    # run_gui()


