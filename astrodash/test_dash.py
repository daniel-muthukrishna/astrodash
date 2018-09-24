import astrodash
import numpy as np

wave = np.arange(2500, 10000, 2)
flux = np.ones(len(wave))
filename = np.array([wave, flux])
# filename = open('/Users/danmuth/PycharmProjects/DASH/templates/superfit_templates/sne/Ib/sn1983N.p12.dat', 'r')
# filename = 'osc-SN2002er-10'
redshift = 0

classification = astrodash.Classify([filename], [redshift], classifyHost=False, knownZ=True, smooth=6, rlapScores=True)
bestFits, redshifts, bestTypes, rlapFlag, matchesFlag, redshiftErrs = classification.list_best_matches(n=5)

print(bestFits)

classification.plot_with_gui(indexToPlot=0)
