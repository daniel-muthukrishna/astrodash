from scipy.fftpack import fft
from scipy.signal import argrelmax
import numpy as np
from scipy.stats import chisquare, pearsonr
from astrodash.restore_model import get_training_parameters
from astrodash.array_tools import mean_zero_spectra
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


class RlapCalc(object):
    def __init__(self, inputFlux, templateFluxes, templateNames, wave, inputMinMaxIndex, templateMinMaxIndexes):
        self.templateFluxes = templateFluxes
        self.templateNames = templateNames
        self.wave = wave
        pars = get_training_parameters()
        w0, w1, self.nw, = pars['w0'], pars['w1'], pars['nw']
        self.inputFlux = mean_zero_spectra(inputFlux, inputMinMaxIndex[0], inputMinMaxIndex[1], self.nw)
        self.templateMinMaxIndexes = templateMinMaxIndexes

        self.dwlog = np.log(w1 / w0) / self.nw

    def _cross_correlation(self, templateFlux, templateMinMaxIndex):
        templateFlux = mean_zero_spectra(templateFlux, templateMinMaxIndex[0], templateMinMaxIndex[1], self.nw)
        inputfourier = fft(self.inputFlux)
        tempfourier = fft(templateFlux)

        product = inputfourier * np.conj(tempfourier)
        xCorr = fft(product)

        rmsInput = np.std(inputfourier)
        rmsTemp = np.std(tempfourier)

        xCorrNorm = (1. / (self.nw * rmsInput * rmsTemp)) * xCorr

        rmsXCorr = np.std(product)

        xCorrNormRearranged = np.concatenate((xCorrNorm[int(len(xCorrNorm) / 2):], xCorrNorm[0:int(len(xCorrNorm) / 2)]))

        #
        # w0 = 2500.  # wavelength range in Angstroms
        # w1 = 11000.
        # nw = 1024.  # number of wavelength bins
        # dwlog = np.log(w1 / w0) / self.nw
        # zaxisindex1 = np.concatenate((np.arange(-self.nw / 2, 0), np.arange(0, self.nw / 2)))
        # zaxis1 = np.zeros(self.nw)
        # zaxis1[0:(self.nw/2 -1)] = (np.exp(abs(zaxisindex1[0:(self.nw/2-1)]) * dwlog) - 1)
        # zaxis1[self.nw/2:] = -(np.exp(abs(zaxisindex1[(self.nw/2):]) * dwlog) - 1)
        # plt.plot(zaxis1, xCorrNormRearranged)
        # plt.show()
        # plt.plot(self.zAxis, np.correlate(self.inputFlux, templateFlux, mode='Full')[::-1][512:1536]/max(np.correlate(self.inputFlux, templateFlux, mode='Full')))
        # plt.plot(self.zAxis, np.correlate(templateFlux, templateFlux, mode='Full')[::-1][512-shift:1536-shift]/max(np.correlate(templateFlux, templateFlux, mode='Full')))

        crossCorr = np.correlate(self.inputFlux, templateFlux, mode='Full')[::-1][int(self.nw/2):int(self.nw + self.nw/2)]/max(np.correlate(self.inputFlux, templateFlux, mode='Full'))

        try:
            deltapeak, h = self._get_peaks(crossCorr)[0]
            shift = int(deltapeak - self.nw / 2)
            autoCorr = np.correlate(templateFlux, templateFlux, mode='Full')[::-1][int(self.nw/2)-shift:int(self.nw + self.nw/2)-shift]/max(np.correlate(templateFlux, templateFlux, mode='Full'))

            aRandomFunction = crossCorr - autoCorr
            rmsA = np.std(aRandomFunction)
        except IndexError as err:
            print("Error: Cross-correlation is zero, probably caused by empty spectrum.", err)
            rmsA = 1

        return xCorr, rmsInput, rmsTemp, xCorrNorm, rmsXCorr, xCorrNormRearranged, rmsA

    def _get_peaks(self, crosscorr):
        peakindexes = argrelmax(crosscorr)[0]

        ypeaks = []
        for i in peakindexes:
            ypeaks.append(abs(crosscorr[i]))

        arr = list(zip(*[peakindexes, ypeaks]))
        arr.sort(key=lambda x: x[1])
        sortedPeaks = list(reversed(arr))

        return sortedPeaks


    def _calculate_r(self, crosscorr, rmsA):
        deltapeak1, h1 = self._get_peaks(crosscorr)[0]  # deltapeak = np.argmax(abs(crosscorr))
        deltapeak2, h2 = self._get_peaks(crosscorr)[1]
        # h = crosscorr[deltapeak]

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
        r = abs((h1 -rmsA) / (np.sqrt(2) * rmsA))
        fom = (h1 - 0.05) ** 0.75 * (h1 / h2)

        return r, deltapeak1, fom

    def get_redshift_axis(self, nw, dwlog):
        zAxisIndex = np.concatenate((np.arange(-nw / 2, 0), np.arange(0, nw / 2)))
        zAxis = np.zeros(nw)
        zAxis[0:int(nw / 2 - 1)] = -(np.exp(abs(zAxisIndex[0:int(nw / 2 - 1)]) * dwlog) - 1)
        zAxis[int(nw / 2):] = (np.exp(abs(zAxisIndex[int(nw / 2):]) * dwlog) - 1)
        zAxis = zAxis[::-1]

        return zAxis

    def calculate_rlap(self, crosscorr, rmsAntisymmetric, templateFlux):
        r, deltapeak, fom = self._calculate_r(crosscorr, rmsAntisymmetric)
        shift = int(deltapeak - self.nw / 2)  # shift from redshift

        # lap value
        iminindex, imaxindex = self.min_max_index(self.inputFlux)
        tminindex, tmaxindex = self.min_max_index(templateFlux)

        overlapminindex = int(max(iminindex + shift, tminindex))
        overlapmaxindex = int(min(imaxindex - 1 + shift, tmaxindex - 1))

        minWaveOverlap = self.wave[overlapminindex]
        maxWaveOverlap = self.wave[overlapmaxindex]

        lap = np.log(maxWaveOverlap / minWaveOverlap)
        rlap = 5 * r * lap

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

    def rlap_score(self, tempIndex):
        xcorr, rmsinput, rmstemp, xcorrnorm, rmsxcorr, xcorrnormRearranged, rmsA = self._cross_correlation(
            self.templateFluxes[tempIndex].astype('float'), self.templateMinMaxIndexes[tempIndex])
        crosscorr = xcorrnormRearranged
        r, lap, rlap, fom = self.calculate_rlap(crosscorr, rmsA, self.templateFluxes[tempIndex])

        return r, lap, rlap, fom

    def rlap_label(self):
        if not np.any(self.inputFlux):
            return "No flux", True

        self.zAxis = self.get_redshift_axis(self.nw, self.dwlog)
        rlapList = []
        for i in range(len(self.templateNames)):
            r, lap, rlap, fom = self.rlap_score(tempIndex=i)
            rlapList.append(rlap)
            if i > 20:
                break

        rlapMean = round(np.mean(rlapList),2)
        rlapLabel = str(rlapMean)

        if rlapMean < 6:
            rlapWarning = True
        else:
            rlapWarning = False

        return rlapLabel, rlapWarning

