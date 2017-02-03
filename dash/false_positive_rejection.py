from scipy.fftpack import fft

class FalsePositiveRejection(object):
    def __init__(self, inputFlux, templateFlux):
        self.inputFlux = inputFlux
        self.templateFlux = templateFlux

    def _cross_correlation(self):
        inputfourier = fft(inputflux)
        tempfourier = fft(tempflux)
        kinput = 2 * np.pi / inputwave
        ktemp = 2 * np.pi / tempwave

        product = inputfourier * np.conj(tempfourier)
        xcorr = fft(product)

        rmsinput = np.std(inputfourier)
        rmstemp = np.std(tempfourier)

        xcorrnorm = (1. / (N * rmsinput * rmstemp)) * xcorr

        rmsxcorr = np.std(product)

        xcorrnormRearranged = np.concatenate((xcorrnorm[len(xcorrnorm) / 2:], xcorrnorm[0:len(xcorrnorm) / 2]))

        return xcorr, rmsinput, rmstemp, xcorrnorm, rmsxcorr, xcorrnormRearranged

    def _get_peaks(self):
        peakindexes = argrelmax(crosscorr)[0]

        ypeaks = []
        for i in peakindexes:
            ypeaks.append(abs(crosscorr[i]))

        arr = zip(*[peakindexes, ypeaks])
        arr.sort(key=lambda x: x[1])
        sortedPeaks = list(reversed(arr))

        return sortedPeaks


    def _calculate_r(self):
        deltapeak1, h1 = get_peaks(crosscorr)[0]  # deltapeak = np.argmax(abs(crosscorr))
        deltapeak2, h2 = get_peaks(crosscorr)[1]
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

    def _calculate_lap(self):
        pass

    def calculate_rlap(self):
        r, deltapeak, fom = r_value(crosscorr)
        shift = deltapeak - N / 2  # shift from redshift

        # lap value
        iminindex, imaxindex = (0, nw)
        zeros = np.where(inputflux == 0)[0]
        j = 0
        for i in zeros:
            if (i != j):
                iminindex = j
                imaxindex = i
                break
            j += 1
        j = int(nw) - 1
        for i in zeros[::-1]:
            if (i != j):
                imaxindex = j
                break
            j += 1

        overlapminindex = max(iminindex + shift, tminindex)
        overlapmaxindex = min(imaxindex - 1 + shift, tmaxindex - 1)

        lap = np.log(inputwave[overlapmaxindex] / inputwave[overlapminindex])
        rlap = r * lap

        fom = fom * lap
        return r, lap, rlap, fom


def __name__ == '__main__':
    pass