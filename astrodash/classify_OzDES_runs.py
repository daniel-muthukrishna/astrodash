import os
import astrodash


def read_ozdes_wiki_atel(atelTextFile):
    """ Read ATEL wiki text, which has been saved to a text file: atelTextFile. 
    Returns the names, redshifts, and wikiClassifications"""
    names = []
    redshifts = []
    wikiClassifications = []
    with open(atelTextFile, 'r') as f:
        for line in f:
            if line[0:3] == 'DES':
                objectInfo = line.split()
                name = objectInfo[0]
                redshift = objectInfo[1]
                wikiClassification = ' '.join(objectInfo[2:])

                try:
                    if redshift != '?':  #  and 'SN' not in wikiClassification:
                        names.append(name)
                        redshifts.append(float(redshift))
                        wikiClassifications.append(wikiClassification)
                except ValueError as e:
                    print("Invalid redshift for line: {0}".format(line))
                    raise e

    return names, redshifts, wikiClassifications


def main(runDirectory, atelTextFile, saveMatchesFilename):

    # Store all the file paths to the objects in this run
    directoryPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), runDirectory)
    allFilePaths = []
    for dirpath, dirnames, filenames in os.walk(directoryPath):
        for filename in [f for f in filenames if f.endswith(".dat")]:
            allFilePaths.append(os.path.join(dirpath, filename))
    allFilePaths.reverse()

    # Get filenames and corresponding redshifts
    names, knownRedshifts, wikiClassifications = read_ozdes_wiki_atel(atelTextFile)
    run = []
    for i in range(len(names)):
        for filePath in allFilePaths:
            if names[i] == filePath.split('/')[-1].split('_')[0]:
                run.append((filePath, knownRedshifts[i], wikiClassifications[i]))
                #break  # Uncomment the break to only classify the last dated spectrum for each object instead of classifying all dates.
    filenames = [i[0] for i in run]
    knownRedshifts = [i[1] for i in run]
    wikiClassifications = [i[2] for i in run]

    # Classify and print the best matches
    classification = astrodash.Classify(filenames, knownRedshifts, classifyHost=False, rlapScores=True, smooth=6)
    bestFits, redshifts, bestTypes, rlapFlag, matchesFlag, redshiftErrs = classification.list_best_matches(n=5, saveFilename=saveMatchesFilename)
    print("{0:17} | {1:5} | {2:8} | {3:10} | {4:6} | {5:10} | {6:10}".format("Name", "  z  ", "DASH_Fit", "  Age ", "Prob.", "Flag", "Wiki Fit"))
    for i in range(len(filenames)):
        print("{0:17} | {1:5} | {2:8} | {3:10} | {4:6} | {5:10} | {6:10}".format('_'.join([filenames[i].split('/')[-1].split('_')[0], filenames[i].split('/')[-1].split('_')[3]]) , redshifts[i], bestTypes[i][0], bestTypes[i][1], bestTypes[i][2], matchesFlag[i].replace(' matches',''), wikiClassifications[i]))

    # Plot one of the matches
    classification.plot_with_gui(indexToPlot=0)


if __name__ == '__main__':
    main(runDirectory='../templates/run034/', atelTextFile='wiki_atel_run034.txt', saveMatchesFilename='DASH_matches_run34.txt')
