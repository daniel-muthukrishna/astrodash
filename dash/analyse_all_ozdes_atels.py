import os
import dash
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

    # Store all the file paths to the objects in this directory
    allFilePaths = os.listdir(spectraDir)

    # Get filenames and corresponding redshifts
    atels = read_all_atels(atelTextFile)
    for filename in allFilePaths:
        name = filename.replace('.fits', '')
        i = np.where(atels['Name'] == name)[0]
        if len(i) > 1:
            print("Something wrong with:", name)
        atels.at[i, 'Filename'] = os.path.join(spectraDir, filename)

    # Check how many observed runs
    count = 0
    for filename in atels.Filename:
        hdulist = afits.open(filename)
        run = hdulist[3].header['SOURCEF'].split('/')[0]
        for i in range(3, len(hdulist)):
            try:
                run_new = hdulist[i].header['SOURCEF'].split('/')[0]
                if run_new != run:
                    count += 1
                    print(count, len(atels.Filename), filename, i, run, run_new)
                    break
            except KeyError:
                pass

    # Classify and print the best matches
    classification = dash.Classify(atels.Filename.values, atels.Redshift.values, classifyHost=False, rlapScores=True, smooth=6)
    bestFits, redshifts, bestTypes, rlapFlag, matchesFlag = classification.list_best_matches(n=5, saveFilename=saveMatchesFilename)

    print("{0:17} | {1:5} | {2:8} | {3:10} | {4:6} | {5:10} | {6:10}".format("Name", "  z  ", "DASH_Fit", "  Age ", "Prob.", "Flag", "Wiki Fit"))
    for i in range(len(atels)):
        print("{0:17} | {1:5} | {2:8} | {3:10} | {4:6} | {5:10} | {6:10}".format(os.path.basename(atels.Filename[i]), round(atels.Redshift[i], 3), bestTypes[i][0], bestTypes[i][1], bestTypes[i][2], matchesFlag[i].replace(' matches',''), "{} {}".format(atels['Type'][i], atels['Phase'][i])))

    # Plot one of the matches
    classification.plot_with_gui(indexToPlot=7)


if __name__=='__main__':
    main(spectraDir='/Users/danmuth/PycharmProjects/DASH/templates/OzDES_data/180413/',
         atelTextFile='/Users/danmuth/PycharmProjects/DASH/templates/OzDES_data/all_atels.txt',
         saveMatchesFilename='DASH_matches_all_atels.txt')

