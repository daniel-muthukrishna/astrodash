from scipy.fftpack import fft
from scipy.signal import argrelmax
import numpy as np
from scipy.stats import chisquare, pearsonr
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


def combined_prob(bestMatchList):
    hostName, prevName, age, prob = bestMatchList[0]
    probInitial = float(prob)
    bestName = prevName
    prevBroadTypeName = prevName[0:2]
    prevMinAge, prevMaxAge = age.split(' to ')
    probTotal = 0.
    agesList = [int(prevMinAge), int(prevMaxAge)]
    probPossible = 0.
    agesListPossible = []
    probPossible2 = 0.
    agesListPossible2 = []
    index = 0
    for host, name, age, prob in bestMatchList[0:10]:
        index += 1
        minAge, maxAge = list(map(int, age.split(' to ')))
        if "IIb" in name:
            broadTypeName = "Ib"
        else:
            broadTypeName = name[0:2]
        if name == prevName:
            if probPossible == 0:
                if (minAge in agesList) or (maxAge in agesList):
                    probTotal += float(prob)
                    prevName = name
                    agesList = agesList + [minAge, maxAge]
                else:
                    probPossible = float(prob)
                    agesListPossible = [minAge, maxAge]
            else:
                if probPossible2 == 0:
                    if ((minAge in agesListPossible) or (maxAge in agesListPossible)) and (
                                (minAge in agesList) or (maxAge in agesList)):
                        probTotal += probPossible + float(prob)
                        agesList = agesList + agesListPossible + [minAge, maxAge]
                        probPossible = 0
                        agesListPossible = []
                    else:
                        probPossible2 = probPossible + float(prob)
                        agesListPossible2 = agesListPossible + [minAge, maxAge]
                else:
                    if ((minAge in agesListPossible2) or (maxAge in agesListPossible2)) and (
                                (minAge in (agesList + agesListPossible)) or (maxAge in (agesList + agesListPossible))):
                        probTotal += probPossible2 + float(prob)
                        agesList = agesList + agesListPossible2 + [minAge, maxAge]
                        probPossible, probPossible2 = 0, 0
                        agesListPossible, agesListPossible2 = [], []

        elif broadTypeName == prevBroadTypeName:
            if probPossible == 0:
                if (minAge in agesList) or (maxAge in agesList):
                    if index <= 2:
                        bestName = broadTypeName
                    probTotal += float(prob)
                    agesList = agesList + [minAge, maxAge]
                else:
                    probPossible = float(prob)
                    agesListPossible = [minAge, maxAge]
            else:
                if ((minAge in agesListPossible) or (maxAge in agesListPossible)) and (
                            (minAge in agesList) or (maxAge in agesList)):
                    probTotal += probPossible + float(prob)
                    agesList = agesList + agesListPossible + [minAge, maxAge]
                    probPossible = 0
                    agesListPossible = []
                else:
                    break
        else:
            break

    bestAge = '%d to %d' % (min(agesList), max(agesList))

    if probTotal > probInitial:
        reliableFlag = True
    else:
        reliableFlag = False

    return hostName, bestName, bestAge, round(probTotal, 4), reliableFlag


class FalsePositiveRejection(object):
    def __init__(self, inputFlux, templateFluxes):
        self.inputFlux = inputFlux
        self.templateFluxes = templateFluxes
        self.nw = len(self.inputFlux)

    def _cross_correlation(self, templateFlux):
        inputfourier = fft(self.inputFlux, self.nw)
        tempfourier = fft(templateFlux, self.nw)

        product = inputfourier * np.conj(tempfourier)
        xcorr = fft(product)

        rmsinput = np.std(inputfourier)
        rmstemp = np.std(tempfourier)

        xcorrnorm = (1. / (self.nw * rmsinput * rmstemp)) * xcorr

        rmsxcorr = np.std(product)

        xcorrnormRearranged = np.concatenate((xcorrnorm[int(len(xcorrnorm) / 2):], xcorrnorm[0:int(len(xcorrnorm) / 2)]))

        #
        # w0 = 2500.  # wavelength range in Angstroms
        # w1 = 11000.
        # nw = 1024.  # number of wavelength bins
        # dwlog = np.log(w1 / w0) / self.nw
        # zaxisindex1 = np.concatenate((np.arange(-self.nw / 2, 0), np.arange(0, self.nw / 2)))
        # zaxis1 = np.zeros(self.nw)
        # zaxis1[0:(self.nw/2 -1)] = (np.exp(abs(zaxisindex1[0:(self.nw/2-1)]) * dwlog) - 1)
        # zaxis1[self.nw/2:] = -(np.exp(abs(zaxisindex1[(self.nw/2):]) * dwlog) - 1)
        # plt.plot(zaxis1, xcorrnormRearranged)
        # plt.show()

        return xcorr, rmsinput, rmstemp, xcorrnorm, rmsxcorr, xcorrnormRearranged

    def _get_peaks(self, crosscorr):
        peakindexes = argrelmax(crosscorr)[0]

        ypeaks = []
        for i in peakindexes:
            ypeaks.append(abs(crosscorr[i]))

        arr = list(zip(*[peakindexes, ypeaks]))
        arr.sort(key=lambda x: x[1])
        sortedPeaks = list(reversed(arr))

        return sortedPeaks


    def _calculate_r(self, crosscorr):
        deltapeak1, h1 = self._get_peaks(crosscorr)[0]  # deltapeak = np.argmax(abs(crosscorr))
        deltapeak2, h2 = self._get_peaks(crosscorr)[1]
        # h = crosscorr[deltapeak]
        rmsxcorr = np.std(crosscorr)
        ##    shift = deltapeak
        ##    arms = 0
        ##    srms = 0
        ##    for k in range(0,int(nw-1)):
        ##        angle = -2*np.pi*k/nw
        ##        phase = complex(np.cos(angle),np.sin(angle))
        ####	f=2
        ####	if (k < k2):
        ####		arg = np.pi * float(k-k1)/float(k2-k1)
        ####		f = f * .25 * (1-np.cos(arg)) * (1-np.cos(arg))
        ####	elif (k > k3):
        ####		arg = np.pi * float(k-k3)/float(k4-k3)
        ####		f = f * .25 * (1+np.cos(arg)) * (1+np.cos(arg))
        ##        arms = arms + (phase*crosscorr[k+1]).imag*(phase*crosscorr[k+1]).imag
        ##        srms = srms + (phase*crosscorr[k+1]).real*(phase*crosscorr[k+1]).real
        ##
        ##    arms = np.sqrt(arms)/nw
        r = abs(h1 / (np.sqrt(2) * rmsxcorr))
        fom = (h1 - 0.05) ** 0.75 * (h1 / h2)

        return r, deltapeak1, fom

    def calculate_rlap(self, crosscorr, templateFlux):
        r, deltapeak, fom = self._calculate_r(crosscorr)
        shift = deltapeak - self.nw / 2  # shift from redshift

        # lap value
        iminindex, imaxindex = self.min_max_index(self.inputFlux)
        tminindex, tmaxindex = self.min_max_index(templateFlux)

        overlapminindex = max(iminindex + shift, tminindex)
        overlapmaxindex = min(imaxindex - 1 + shift, tmaxindex - 1)

        lap = np.log(overlapmaxindex / overlapminindex)
        rlap = r * lap


        fom = fom * lap
        # print r, lap, rlap, fom
        return r, lap, rlap, fom

    def min_max_index(self, flux):
        minindex, maxindex = (0, self.nw - 1)
        zeros = np.where(flux == 0)[0]
        j = 0
        for i in zeros:
            if (i != j):
                break
            j += 1
            minindex = j
        j = int(self.nw) - 1
        for i in zeros[::-1]:
            if (i != j):
                break
            j -= 1
            maxindex = j

        return minindex, maxindex

    def calculate_chi_squared(self, templateFlux):
        """ Only calculate on overlap region (MAYBE AVERAGE ALL TEMPLATES LATER)"""
        iminindex, imaxindex = self.min_max_index(self.inputFlux)
        tminindex, tmaxindex = self.min_max_index(templateFlux)

        overlapminindex = max(iminindex, tminindex)
        overlapmaxindex = min(imaxindex - 1, tmaxindex - 1)

        inputSpecOverlapped = 100 * (1+self.inputFlux[overlapminindex:overlapmaxindex])
        templateSpecOverlapped = 100 * (1+templateFlux[overlapminindex:overlapmaxindex])


        chi2 = chisquare(inputSpecOverlapped, templateSpecOverlapped)[0]
        pearsonCorr = pearsonr(inputSpecOverlapped, templateSpecOverlapped)

        return chi2, pearsonCorr

    def rejection_label(self):
        chi2List = []
        pearsonList = []
        for templateFlux in self.templateFluxes:
            chi2 = self.calculate_chi_squared(templateFlux)[0]
            chi2List.append(chi2)
            pearson = self.calculate_chi_squared(templateFlux)[1]
            pearsonList.append(pearson)

        chi2Mean = round(np.mean(chi2List),2)
        pearsonMean = np.mean(pearsonList)
        print(chi2Mean, np.median(chi2List), min(chi2List), max(chi2List), len(chi2List))
        print(pearsonMean, np.median(pearsonList), min(pearsonList), max(pearsonList), len(pearsonList))

        return "%s, Pearson=%s" % (str(chi2Mean), str(pearsonMean))

    def rejection_label2(self):
        rlapList = []
        for templateFlux in self.templateFluxes:
            xcorr, rmsinput, rmstemp, xcorrnorm, rmsxcorr, xcorrnormRearranged = self._cross_correlation(templateFlux.astype('float'))
            crosscorr = xcorrnormRearranged
            r, lap, rlap, fom = self.calculate_rlap(crosscorr, templateFlux)
            rlapList.append(rlap)

        rlapMean = round(np.mean(rlapList),2)
        print(rlapMean, np.median(rlapList), min(rlapList), max(rlapList))

        return str(rlapMean)

