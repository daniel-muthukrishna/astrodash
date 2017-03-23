import pickle

parameters = {
    'typeList': ['Ia-norm', 'Ia-91T', 'Ia-91bg', 'Ia-csm', 'Ia-02cx', 'Ia-pec',
                'Ib-norm', 'Ibn', 'IIb', 'Ib-pec', 'Ic-norm', 'Ic-broad',
                'Ic-pec', 'IIP', 'IIL', 'IIn', 'II-pec'],
    'nTypes': 17,
    'w0': 2500.,             #wavelength range in Angstroms
    'w1': 10000.,
    'nw': 1024,             #number of wavelength bins
    'minAge': -20.,
    'maxAge': 50.,
    'ageBinSize': 4.
}




# Saving the objects:
with open('training_params_v02.pickle', 'wb') as f:
    pickle.dump(parameters, f, protocol=2)

# Getting back the objects:
with open('training_params_v02.pickle', 'rb') as f:
    pars = pickle.load(f)
nTypes, w0, w1, nw, minAge, maxAge, ageBinSize, typeList = pars['nTypes'], pars['w0'], pars['w1'], pars['nw'], \
                                                           pars['minAge'], pars['maxAge'], pars['ageBinSize'], \
                                                           pars['typeList']

