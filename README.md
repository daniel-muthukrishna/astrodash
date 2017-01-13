# DASH
Supernovae classifying and redshifting software: development stage


## 1. How to install:

    1.1 pip install astrodash

        or download from github (https://github.com/daniel-muthukrishna/DASH)

## 2. Get started with the Python Library interface:
    2.1 Use the following example code:
        import dash
        classification = dash.Classify([filenames], [knownRedshifts])
        print classification.list_best_matches(n=1)

    2.2 To open the gui from a script use:
        import dash
        dash.run_gui()

## 3. Get started with GUI
    2.1 Run GUI/main.py

    2.2 Once open, type in a known redshift

    2.3 Browse for any single spectrum FITS, ASCII, dat, or two-column text file.

    2.4 Click any of the best matches to view the continuum-subtracted binned spectra.

    2.5 If the input spectrum is too noisy, increase the smoothing level, and click 'Re-fit with priors'


## 4. Dependencies:
    Using pip will automatically install numpy, scipy, specutils, pyqtgraph, and tensorflow.

    PyQt4

        This can be installed with anaconda: "conda install pyqt=4" (or else independently - only needed for the GUI)

## 5. How to raise issues:

## 6. Examples

## 7. API Usage
Notes:
    Current version requires an input redshift (inaccurate results if redshift is unknown)


