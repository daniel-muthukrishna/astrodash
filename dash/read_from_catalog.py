import numpy as np
from collections import OrderedDict
import json
from six.moves.urllib.request import urlopen


def read_osc_input(filename):
    """ self.filename in the form osc-name-ageIndex. E.g. osc-sn2002er-10"""
    osc, objName, ageIdx = filename.split('-')
    url = "https://api.sne.space/" + objName + "/spectra/time+data?item={0}".format(ageIdx)
    response = urlopen(url)
    data = json.loads(response.read(), object_pairs_hook=OrderedDict)

    data = data[next(iter(data))]['spectra'][0][1]
    data = np.array(list(map(list, zip(*data)))).astype(np.float)
    wave, flux = data[0], data[1]

    url = "https://api.sne.space/" + objName + "/redshift/value"
    response = urlopen(url)
    redshift = json.loads(response.read(), object_pairs_hook=OrderedDict)
    redshift = float(redshift[next(iter(redshift))]['redshift'][0][0])

    return wave, flux, redshift

# filename must start with have catalog key frollowed by '-'. E.g. osc-OTHERINFO
catalogDict = {'osc': read_osc_input}
