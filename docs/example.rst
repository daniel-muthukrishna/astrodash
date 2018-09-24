=======
Example
=======

Example script classifying some spectra from the Open Supernova Catalog and some from OzDES ATEL9570:

This example automatically classifies 4 spectra. The last line plots the second spectrum on the GUI.

.. code-block:: python

    import astrodash

    example = [
        ('osc-sn2002er-10', 0.0),
        ('osc-sn2013fs-8', 0.0),
        ('DES16C3bq_C3_combined_160925_v10_b00.dat', 0.237),
        ('DES16E2aoh_E2_combined_160925_v10_b00.dat', 0.403)]

    # Create filenames and knownRedshifts lists
    filenames = [i[0] for i in example]
    knownRedshifts = [i[1] for i in example]

    # Classify all spectra
    classification = astrodash.Classify(filenames, knownRedshifts, classifyHost=False, knownZ=True, smooth=6)
    bestFits, redshifts, bestTypes, rlapFlag, matchesFlag = classification.list_best_matches(n=5, saveFilename='example_best_fits.txt')

    # Plot sn2002ey from open supernova catalog (2nd spectrum)
    classification.plot_with_gui(indexToPlot=1)
