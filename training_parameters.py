import pickle


typeList = ['Ia-norm', 'Ia-91T', 'Ia-91bg', 'Ia-csm', 'Ia-02cx', 'Ia-pec',
            'Ib-norm', 'Ibn', 'IIb', 'Ib-pec', 'Ic-norm', 'Ic-broad',
            'Ic-pec', 'IIP', 'IIL', 'IIn', 'II-pec']

nTypes = len(typeList)
w0 = 2500. #wavelength range in Angstroms
w1 = 10000.
nw = 1024. #number of wavelength bins
minAge = -20.
maxAge = 50.
ageBinSize = 4.


# Saving the objects:
with open('training_params.pickle', 'w') as f:
    pickle.dump([nTypes, w0, w1, nw, minAge, maxAge, ageBinSize, typeList], f)

# Getting back the objects:
with open('training_params.pickle') as f:
    nTypes, w0, w1, nw, minAge, maxAge, ageBinSize, typeList = pickle.load(f)
