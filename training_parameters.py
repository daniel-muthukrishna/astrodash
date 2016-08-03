import pickle


typeList = ['Ia-norm', 'IIb', 'Ia-pec', 'Ic-broad', 'Ia-csm', 'Ic-norm', 'IIP', 'Ib-pec',
                  'IIL', 'Ib-norm', 'Ia-91bg', 'II-pec', 'Ia-91T', 'IIn']
nTypes = len(typeList)
w0 = 2500. #wavelength range in Angstroms
w1 = 11000.
nw = 1024. #number of wavelength bins
minAge = -50
maxAge = 50
ageBinSize = 4.


# Saving the objects:
with open('training_params.pickle', 'w') as f:
    pickle.dump([nTypes, w0, w1, nw, minAge, maxAge, ageBinSize, typeList], f)

# Getting back the objects:
with open('training_params.pickle') as f:
    nTypes, w0, w1, nw, minAge, maxAge, ageBinSize, typeList = pickle.load(f)
