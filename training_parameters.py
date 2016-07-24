import pickle

nTypes = 14
w0 = 2500. #wavelength range in Angstroms
w1 = 11000.
nw = 1024. #number of wavelength bins
minAge = -50
maxAge = 50
ageBinSize = 4.

# Saving the objects:
with open('training_params.pickle', 'w') as f:
    pickle.dump([nTypes, w0, w1, nw, minAge, maxAge, ageBinSize], f)

# Getting back the objects:
with open('training_params.pickle') as f:
    nTypes, w0, w1, nw, minAge, maxAge, ageBinSize = pickle.load(f)
