# DASH
Supernovae classifying and redshifting software: development stage


## 1. How to install:

    1.1 pip install astrodash

        or download from github (https://github.com/daniel-muthukrishna/DASH)

## 2. Get started with the Python Library interface:
    2.1 Use the following example code:
        import dash
        classification = dash.Classify([filenames], [knownRedshifts])
        print(classification.list_best_matches(n=1))  # Shows top 'n' matches for each spectrum

    2.2 To open the gui from a script use:
        import dash
        dash.run_gui()


## 3. Get started with GUI
    2.1 Run gui_main.py

    2.2 Once open, type in a known redshift

    2.3 Browse for any single spectrum FITS, ASCII, dat, or two-column text file.

    2.4 Click any of the best matches to view the continuum-subtracted binned spectra.

    2.5 If the input spectrum is too noisy, increase the smoothing level, and click 'Re-fit with priors'


## 4. Dependencies:
    Using pip will automatically install numpy, scipy, specutils, pyqtgraph, and tensorflow.

    PyQt5

        PyQt5 is only needed if you would like to use a graphical interface. It is not available on pip.
        It can be installed with anaconda:
            "conda install pyqt"

## 5. Platforms
    5.1 Mac/Unix
        DASH is available on both Python2 and Python3 distributions. It can easily be installed with
            pip install astrodash

    5.2 Windows
        Currently one of the primary dependencies, Tensorflow, is only available on Python 3 for Windows.
        So DASH is available on Python3 distributions. It can be installed with:
            pip install astrodash
        If this fails, try first installing specutils with the following:
            conda install -c astropy specutils


## 6. Example Usage
    6.1 Example from OzDES Run025/ATEL9570:
        This example automatically classifies 10 spectra. The last line plots the fifth spectrum on the GUI.
        ```
        import dash

        atel9570 = [
            ('DES16C3bq_C3_combined_160925_v10_b00.dat', 0.237),
            ('DES16E2aoh_E2_combined_160925_v10_b00.dat', 0.403),
            ('DES16X3aqd_X3_combined_160925_v10_b00.dat', 0.033),
            ('DES16X3biz_X3_combined_160925_v10_b00.dat', 0.24),
            ('DES16C2aiy_C2_combined_160926_v10_b00.dat', 0.182),
            ('DES16C2ma_C2_combined_160926_v10_b00.dat', 0.24),
            ('DES16X1ge_X1_combined_160926_v10_b00.dat', 0.25),
            ('DES16X2auj_X2_combined_160927_v10_b00.dat', 0.144),
            ('DES16E2bkg_E2_combined_161005_v10_b00.dat', 0.478),
            ('DES16E2bht_E2_combined_161005_v10_b00.dat', 0.392)]
    
        filenames = [i[0]) for i in atel9570]
        knownRedshifts = [i[1] for i in atel9570]

        classification = dash.Classify(filenames, knownRedshifts)
        print(classification.list_best_matches(n=3))
        classification.plot_with_gui(indexToPlot=5)
        ```

## 7. API Usage
Notes:
    Current version requires an input redshift (inaccurate results if redshift is unknown)



