import numpy as np
from collections import OrderedDict
import json
from six.moves.urllib.request import urlopen
from astropy.time import Time


def read_osc_input(filename, template=False):
    """ self.filename in the form osc-name-ageIndex. E.g. osc-sn2002er-10
    TODO: Do error checking if input is invalid
    """
    osc, objName, ageIdx = filename.split('-')

    def read_json(url):
        response = urlopen(url)
        return json.loads(response.read(), object_pairs_hook=OrderedDict)

    # Redshift
    urlRedshift = "https://api.sne.space/" + objName + "/redshift/value"
    redshift = read_json(urlRedshift)
    redshift = float(redshift[next(iter(redshift))]['redshift'][0][0])

    if template is False:
        # Spectrum
        urlSpectrum = "https://api.sne.space/" + objName + "/spectra/time+data?item={0}".format(ageIdx)
        data = read_json(urlSpectrum)
        data = data[next(iter(data))]['spectra'][0][1]
        data = np.array(list(map(list, zip(*data)))).astype(np.float)
        if data.shape[0] == 3:
            wave, flux, fluxerr = data
        elif data.shape[0] == 2:
            wave, flux = data
        else:
            raise Exception("Error reading the given OSC input: {}. Check data at {}".format(filename, urlSpectrum))

        return wave, flux, redshift

    elif template is True:
        # Age Max
        urlAgeMax = "https://api.sne.space/" + objName + "/maxdate/value"
        ageMax = read_json(urlAgeMax)
        ageMax = ageMax[next(iter(ageMax))]['maxdate'][0][0]
        ageMax = Time(ageMax.replace('/','-')).mjd

        # Type
        urlTType = "https://api.sne.space/" + objName + "/claimedtype/value"
        tType = read_json(urlTType)
        tType = tType[next(iter(tType))]['claimedtype']  # List of types... choose one [0][0]

        # Spectrum
        urlSpectrum = "https://api.sne.space/" + objName + "/spectra/time+data"
        data = read_json(urlSpectrum)
        data = data[next(iter(data))]['spectra']
        nCols = len(data)  # number of ages

        waves = []
        fluxes = []
        ages = []
        for datum in data:
            age, spectrum = datum
            age = age - ageMax
            ages.append(age)

            wave, flux = np.array(list(map(list, zip(*data)))).astype(np.float)
            wave = wave / (redshift + 1)  # De-redshift spectrum
            waves.append(wave)
            fluxes.append(flux)

        return waves, fluxes, nCols, ages, tType


# filename must start with catalog key followed by '-'. E.g. osc-OTHERINFO
catalogDict = {'osc': read_osc_input}


if __name__ == '__main__':
    read_osc_input('osc-sn2002er-10')
