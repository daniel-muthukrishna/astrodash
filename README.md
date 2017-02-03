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

    PyQt4

        PyQt4 is only needed if you would like to use a graphical interface. It is not available on pip.
        It can be installed with anaconda:
            "conda install pyqt=4"

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
    6.1 Example from OzDES Run028:
        This example automatically classifies 11 spectra. The last line plots the first spectrum on the GUI.
        ```
        import dash

        atel9742 = [
            ('DES16E1ciy_E1_combined_161101_v10_b00.dat', 0.174),
            ('DES16S1cps_S1_combined_161101_v10_b00.dat', 0.274),
            ('DES16E2crb_E2_combined_161102_v10_b00.dat', 0.229),
            ('DES16E2clk_E2_combined_161102_v10_b00.dat', 0.367),
            ('DES16E2cqq_E2_combined_161102_v10_b00.dat', 0.426),
            ('DES16X2ceg_X2_combined_161103_v10_b00.dat', 0.335),
            ('DES16X2bkr_X2_combined_161103_v10_b00.dat', 0.159),
            ('DES16X2crr_X2_combined_161103_v10_b00.dat', 0.312),
            ('DES16X2cpn_X2_combined_161103_v10_b00.dat', 0.28),
            ('DES16X2bvf_X2_combined_161103_v10_b00.dat', 0.135),
            ('DES16C1cbg_C1_combined_161103_v10_b00.dat', 0.111),
            ('DES16C2cbv_C2_combined_161103_v10_b00.dat', 0.109),
            ('DES16C1bnt_C1_combined_161103_v10_b00.dat', 0.351),
            ('DES16C3at_C3_combined_161031_v10_b00.dat', 0.217),
            ('DES16X3cpl_X3_combined_161031_v10_b00.dat', 0.205),
            ('DES16E2cjg_E2_combined_161102_v10_b00.dat', 0.48),
            ('DES16X2crt_X2_combined_161103_v10_b00.dat', 0.57)]

        filenames = [i[0]) for i in atel9742]
        knownRedshifts = [i[1] for i in atel9742]

        classification = dash.Classify(filenames, knownRedshifts)
        print(classification.list_best_matches(n=3))
        classification.plot_with_gui(indexToPlot=1)
        ```

## 7. API Usage
Notes:
    Current version requires an input redshift (inaccurate results if redshift is unknown)



