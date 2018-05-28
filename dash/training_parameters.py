import pickle
import os


def create_training_params_file(dataDirName):
    parameters = {
        'typeList': ['Ia-norm', 'Ia-91T', 'Ia-91bg', 'Ia-csm', 'Ia-02cx', 'Ia-pec',
                     'Ib-norm', 'Ibn', 'IIb', 'Ib-pec', 'Ic-norm', 'Ic-broad',
                     'Ic-pec', 'IIP', 'IIL', 'IIn', 'II-pec'],
        'nTypes': 17,
        'w0': 3500.,  # wavelength range in Angstroms
        'w1': 10000.,
        'nw': 1024,  # number of wavelength bins
        'minAge': -20.,
        'maxAge': 50.,
        'ageBinSize': 4.,
        'galTypeList': ['E', 'S0', 'Sa', 'Sb', 'Sc', 'SB1', 'SB2', 'SB3', 'SB4', 'SB5', 'SB6']
    }

    trainingParamsFilename = os.path.join(dataDirName, 'training_params.pickle')

    # Saving the objects:
    with open(trainingParamsFilename, 'wb') as f:
        pickle.dump(parameters, f, protocol=2)
    print("Saved files to %s" % trainingParamsFilename)

    # Getting back the objects:
    with open(trainingParamsFilename, 'rb') as f:
        pars = pickle.load(f)
        nTypes, w0, w1, nw, minAge, maxAge, ageBinSize, typeList, galTypeList = pars['nTypes'], pars['w0'], pars['w1'], \
                                                                                pars['nw'], pars['minAge'], \
                                                                                pars['maxAge'], pars['ageBinSize'], \
                                                                                pars['typeList'], pars['galTypeList']

    return trainingParamsFilename


if __name__ == '__main__':
    trainingParamsFilename1 = create_training_params_file('data_files/')
