=======
Example
=======

Example script classifying some spectra from OzDES Run025/ATEL9570:

This example automatically classifies 10 spectra. The last line plots the fifth spectrum on the GUI.

.. code-block:: python

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
        ('DES16E2bht_E2_combined_161005_v10_b00.dat', 0.392)
        ]

    # Create filenames and knownRedshifts lists
    filenames = [i[0] for i in atel9570]
    knownRedshifts = [i[1] for i in atel9570]

    # Classify all spectra
    classification = dash.Classify(filenames, knownRedshifts, classifyHost=False)
    bestFits, redshifts, bestTypes, rlapFlag, matchesFlag = classification.list_best_matches(n=5, saveFilename='ATEL_best_fits.txt')

    # Plot DES16C3bq
    classification.plot_with_gui(indexToPlot=4)
