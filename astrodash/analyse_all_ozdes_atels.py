import os
import glob
import astrodash
import pandas as pd
import numpy as np
import astropy.io.fits as afits


def read_all_atels(atelTextFile):
    """ Read the text file containing all OzDES ATel
    Returns the names, redshifts, type, phase and notes from the ATel
    atels Index(['Name', 'RA (J2000)', 'Dec (J2000)', 'Discovery Date (UT) ',
       'Discovery Mag (r)', 'Spectrum Date (UT)', 'Redshift', 'Type', 'Phase',
       'Notes'],
      dtype='object')
    """

    atels = pd.read_csv(atelTextFile, delimiter='|')
    atels['Name'] = atels['Name'].str.strip()
    pd.to_numeric(atels['Redshift'])
    atels['Filename'] = ''

    return atels


def main(spectraDir, atelTextFile, saveMatchesFilename):
    filenames, knownRedshifts, wikiClassifications = [], [], []

    # Store all the file paths to the objects in this directory
    allFilePaths = glob.glob('%s/*.dat' % spectraDir)

    # Get filenames and corresponding redshifts
    atels = read_all_atels(atelTextFile)
    for i, row in atels.iterrows():
        count = 0
        for filePath in allFilePaths:
            name = os.path.basename(filePath).replace('.dat', '').split('_')[0]
            if row.Name == name:
                filenames.append(filePath)
                knownRedshifts.append(row.Redshift)
                wikiClassifications.append("{} {}".format(row.Type, row.Phase))
                count += 1
                # break  # Uncomment the break to only classify the last dated spectrum for each object instead of classifying all dates.
        print(count)
        if count == 0:
            print(row.Name)

    # Classify and print the best matches
    classification = astrodash.Classify(filenames, knownRedshifts, classifyHost=False, rlapScores=True, smooth=6, knownZ=True, data_files='models_v06')
    bestFits, redshifts, bestTypes, rlapFlag, matchesFlag = classification.list_best_matches(n=5, saveFilename=saveMatchesFilename)

    print("{0:17} | {1:5} | {2:8} | {3:10} | {4:6} | {5:10} | {6:10}".format("Name", "  z  ", "DASH_Fit", "  Age ", "Prob.", "Flag", "Wiki Fit"))
    for i in range(len(filenames)):
        print("{0:17} | {1:5} | {2:8} | {3:10} | {4:6} | {5:10} | {6:10}".format('_'.join([filenames[i].split('/')[-1].split('_')[0], filenames[i].split('/')[-1].split('_')[3]]) , redshifts[i], bestTypes[i][0], bestTypes[i][1], bestTypes[i][2], matchesFlag[i].replace(' matches',''), wikiClassifications[i]))

    # Plot one of the matches
    classification.plot_with_gui(indexToPlot=5)


if __name__=='__main__':
    main(spectraDir='/Users/danmuth/PycharmProjects/astrodash/templates/OzDES_data/transients_all/',
         atelTextFile='/Users/danmuth/PycharmProjects/astrodash/templates/OzDES_data/all_atels.txt',
         saveMatchesFilename='DASH_matches_all_atels.txt')

