=======
Example
=======

Example script classifying some spectra from the Open Supernova Catalog (OSC) and some from OzDES ATel9570:

This example automatically classifies 4 spectra. The last line plots the second spectrum on the GUI. The redshift of the OSC objects is taken from the OSC automatically no matter what redshift value the user inputs.

.. code-block:: python

    import astrodash

    example = [
        ('osc-sn2002er-10', 'osc'),
        ('osc-sn2013fs-8', 'osc'),
        ('DES16C3bq_C3_combined_160925_v10_b00.dat', 0.237),
        ('DES16E2aoh_E2_combined_160925_v10_b00.dat', 0.403)]

    # Create filenames and knownRedshifts lists
    filenames = [i[0] for i in example]
    knownRedshifts = [i[1] for i in example]

    # Classify all spectra
    classification = astrodash.Classify(filenames, knownRedshifts, classifyHost=False, knownZ=True, smooth=6)
    bestFits, redshifts, bestTypes, rlapFlag, matchesFlag = classification.list_best_matches(n=5, saveFilename='example_best_fits.txt')

    # Plot sn2013fs from open supernova catalog (2nd spectrum)
    classification.plot_with_gui(indexToPlot=1)
