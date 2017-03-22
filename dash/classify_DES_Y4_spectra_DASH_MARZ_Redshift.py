import os
import dash

directory = r'/home/daniel/Documents/DES_Y4_Spectra/'

# # LIST ALL SPECTRAL FILENAMES FROM DROPBOX
# filenames = []
# extensions = ('.flm', '.dat', '.txt', '.fits')
# for path, subdirs, files in os.walk(directory):
#     for name in files:
#         if name.lower().endswith(extensions):
#             filenames.append(os.path.join(path, name).replace(directory, ''))
# filenames.sort()
# for f in filenames:
#     print("('%s', 0)," % f)

#Convert files to fits

#Get redshifts from MARZ


filenames = [os.path.join(directory, i[0]) for i in atels]
knownRedshifts = [i[1] for i in atels]

classification = dash.Classify(filenames, knownRedshifts)
bestFits, bestTypes, rejectionLabels = classification.list_best_matches(n=5)

# SAVE BEST MATCHES
print bestFits
f = open('classification_results.txt', 'w')
for i in range(len(atels)):
    f.write("%s   %s     %s      %s\n %s\n\n" % (atels[i][0], atels[i][1], bestTypes[i], rejectionLabels[i], bestFits[i]))
f.close()
print("Finished classifying %d spectra!" % len(atels))

# PLOT SPECTRUM ON GUI
# classification.plot_with_gui(indexToPlot=0)

