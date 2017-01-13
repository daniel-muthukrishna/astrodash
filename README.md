# DASH
Supernovae classifying and redshifting software: development stage


## 1. How to install:

    1.1 pip install astrodash

        or download from github (https://github.com/daniel-muthukrishna/DASH)

## 2. Get started with the Python Library interface:
    2.1 Use the following example code:
        import dash
        classification = dash.Classify([filenames], [knownRedshifts])
        print classification.list_best_matches(n=1)  # Shows top 'n' matches for each spectrum

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

## 6. Example Usage
    6.1 Example from OzDES Run028:
        This example automatically classifies 11 spectra. The last line plots the first spectrum on the GUI.
        ```
        import dash

        filenames = []
        filenames.append('DES16C3elb_C3_combined_161227_v10_b00.dat')
        filenames.append('DES16X3dvb_X3_combined_161225_v10_b00.dat')
        filenames.append('DES16C2ege_C2_combined_161225_v10_b00.dat')
        filenames.append('DES16X3eww_X3_combined_161225_v10_b00.dat')
        filenames.append('DES16X3enk_X3_combined_161225_v10_b00.dat')
        filenames.append('DES16S1ffb_S1_combined_161226_v10_b00.dat')
        filenames.append('DES16C1fgm_C1_combined_161226_v10_b00.dat')
        filenames.append('DES16X2dzz_X2_combined_161226_v10_b00.dat')
        filenames.append('DES16X1few_X1_combined_161227_v10_b00.dat')
        filenames.append('DES16X1chc_X1_combined_161227_v10_b00.dat')
        filenames.append('DES16S2ffk_S2_combined_161227_v10_b00.dat')


        knownRedshifts = []
        knownRedshifts.append(0.429)
        knownRedshifts.append(0.329)
        knownRedshifts.append(0.348)
        knownRedshifts.append(0.445)
        knownRedshifts.append(0.331)
        knownRedshifts.append(0.164)
        knownRedshifts.append(0.361)
        knownRedshifts.append(0.325)
        knownRedshifts.append(0.311)
        knownRedshifts.append(0.043)
        knownRedshifts.append(0.373)

        classification = dash.Classify(filenames, knownRedshifts)
        print classification.list_best_matches(n=3)
        classification.plot_with_gui(indexToPlot=0)
        ```

## 7. API Usage
Notes:
    Current version requires an input redshift (inaccurate results if redshift is unknown)


