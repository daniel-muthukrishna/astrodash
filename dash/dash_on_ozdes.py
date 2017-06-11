import time
t0 = time.time()
import os
import dash
import pandas as pd
import numpy as np
import scipy
import shutil



dfNewList = pd.read_csv('../templates/typeIa_withcuts_Moller090517.csv', sep=',', header=1)
newList = dfNewList.values
cidNew = newList[:, 2]
zNew = newList[:, 43] #[-1]
numTransientsNewList = len(cidNew)

df = pd.read_csv('../templates/DES3Y_SNIa.csv', sep=',', header=1)
ozdesList = df.values
numTransients = len(ozdesList[:,0])
names = ozdesList[:, 1]
fields = ozdesList[:, 2]
snid = ozdesList[:, 0]
for i in range(numTransients):
    ozdesList[i,2] = str(ozdesList[i,2]).replace(",", '&')
redshifts = ozdesList[:,3]
specType = ozdesList[:,4]
ozdesList = np.hstack((ozdesList, np.zeros((numTransients,1))))
print(ozdesList)

for cid in cidNew:
    if cid not in snid:
        print(cid)
        print("CID NOT IN LIST\n\n")

directory1 = '../templates/maipenrai/'  # r'/priv/maipenrai/ozdes/ftp/data/'
directory2 = '../templates/non_aat/'

# LIST ALL SPECTRAL FILENAMES
filenames = []
extensions = ('.dat', '.fits', '.txt')
# This loops goes first so that the files in 'copiedFiles' form maipenrai will replace the downloaded ATC spectra
for path, subdirs, files in os.walk(directory1):
    for name in files:
        if name.lower().endswith(extensions):
            filenames.append(os.path.join(path, name))

for path, subdirs, files in os.walk(directory2):
    for name in files:
        if name.lower().endswith(extensions):
            filenames.append(os.path.join(path, name))

# for f in filenames:
#     print("('%s', 0)," % f)


for i in range(numTransients):
    for f in filenames:
        if names[i] in f:
            ozdesList[i,5] = f # If there is more than one entry, then it will take the last one in the filenames list

print(ozdesList)

names = []
filenames = []
knownRedshifts = []
for (snid, name, field, z, specType, filename) in ozdesList:
    if filename != 0:
        filenames.append(filename)
        if snid in cidNew:
            newListIndex = np.where(cidNew == snid)[0][0]
            # print(z-zNew[newListIndex])
            z = zNew[newListIndex]
        knownRedshifts.append(z)
        names.append(name)
        # shutil.copy2(directory+filename, './copiedFiles')
t1 = time.time()
print("Time spent reading files: {0:.2f}".format(t1 - t0))
classification = dash.Classify(filenames, knownRedshifts, classifyHost=False, smooth=7, knownZ=True)
bestFits, redshifts, bestTypes, rejectionLabels, reliableFlags = classification.list_best_matches(n=5)
t2 = time.time()
print("Time spent classifying: {0:.2f}".format(t2 - t1))
# SAVE BEST MATCHES
print(bestFits)
f = open('classification_results.txt', 'w')
for i in range(len(filenames)):
    f.write("%s   z=%s     %s      %s     %s\n %s\n\n" % (names[i], redshifts[i], bestTypes[i], reliableFlags[i], rejectionLabels[i], bestFits[i]))
f.close()
print("Finished classifying %d spectra!" % len(filenames))

ozdesList = np.hstack((ozdesList, np.zeros((numTransients,11))))
for i in range(numTransients):
    for j in range(len(knownRedshifts)):
        if names[j] in str(ozdesList[i,5]):
            ozdesList[i,7:10] = bestTypes[j][0:3]
            ozdesList[i,10] = str(reliableFlags[j])
            ozdesList[i,11] = ' | '.join(bestFits[j][0])
            ozdesList[i,12] = ' | '.join(bestFits[j][1])
            ozdesList[i,13] = ' | '.join(bestFits[j][2])
            ozdesList[i,14] = ' | '.join(bestFits[j][3])
            ozdesList[i,15] = ' | '.join(bestFits[j][4])
            ozdesList[i, 16] = rejectionLabels[j]  # round(float(rejectionLabels[j]),3)
            snid = ozdesList[i, 0]
            if snid in cidNew:
                newListIndex = np.where(cidNew == snid)[0][0]
                ozdesList[i, 6] = str(zNew[newListIndex])
            else:
                ozdesList[i, 6] = '-'
                # print(ozdesList[i, 1], snid)

print(ozdesList)
# ozdesList = np.delete(ozdesList, [5], axis=1)
#ozdesList1 = ozdesList[:,0:11]
ozdesList2 = ozdesList
print(ozdesList)
print(ozdesList[1])
#ozdesList1 = np.insert(ozdesList1, 0, np.array([['SNID', 'TRANSIENT_NAME', 'FIELD', 'Z_SPEC', 'SPEC_TYPE', 'FILENAME', 'DASH_TYPE', 'DASH_AGE', 'DASH_PROB', 'RELIABLE?', 'RLAP']]), axis=0)
ozdesList2 = np.insert(ozdesList2, 0, np.array([['SNID', 'TRANSIENT_NAME', 'FIELD', 'Z_SPEC', 'SPEC_TYPE', 'FILENAME', 'Z_NEW_LIST', 'DASH_TYPE', 'DASH_AGE', 'DASH_PROB', 'RELIABLE?', 'RANK_1', 'RANK_2', 'RANK_3', 'RANK_4', 'RANK_5', 'REJECTION_SCORES']]), axis=0)

#np.savetxt('DES3Y_SNIa_include_DASH_without_ranking.csv', ozdesList1, delimiter=",", fmt='%s')
np.savetxt('DES3Y_SNIa_include_DASH_withHost_withNoise.csv', ozdesList2, delimiter=",", fmt='%s')

t3 = time.time()
print("Time spent saving output to files: {0:.2f}".format(t3 - t2))
print("Time spent total for {0} spectra (setup + classify + save): {1:.2f}s ... ({2:.2f} + {3:.2f} + {4:.2f})".format(len(filenames), (t3 - t0), (t1 - t0), (t2 - t1), (t3 - t2)))

os.system("open DES3Y_SNIa_include_DASH_withHost_withNoise.csv")
classification.plot_with_gui(indexToPlot=15)
