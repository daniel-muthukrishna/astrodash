# DASH
Supernovae classifying and redshifting software: development stage


1. How to install:

    1.1 Install tensorflow (https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#download-and-setup).

    1.2 pip install deepstars

        or download from github (https://github.com/daniel-muthukrishna/SNClassifying_Pre-alpha)

2. Get started:
    2.1 Run GUI/main.py

    2.2 Once open, type in a known redshift

    2.3 Browse for any single spectrum FITS, ASCII, dat, or two-column text file.

    2.4 Click any of the best matches to view the continuum-subtracted binned spectra.

    2.5 If the input spectrum is too noisy, increase the smoothing level, and click 'Re-fit with priors'


3. Dependencies:
    python2.7 (unconfirmed python3)

    numpy

    scipy

    PyQt4

    tensorflow

3. How to raise issues:

4. Examples

5. API Usage
Notes:
    Current version requires an input redshift (inaccurate results if redshift is unknown)


