import scipy.io
import numpy as np
import pickle

FILE="drug.pickle"

FILE_e="e.pickle"
FILE_gpcr="gpcr.pickle"
FILE_ic="ic.pickle"
FILE_nr="nr.pickle"

mat=scipy.io.loadmat('DtiData.mat')

print()
print()

# for key in sorted(mat.keys()):
# 	if(key[0]!='_' and key[0]!='N'):
# 		print(key, mat[key].shape)
# 		print(np.unique(mat[key

pickle.dump(mat['eAdmatDGC_inl'], open(FILE_e, 'wb'))
pickle.dump(mat['gpcrAdmatDGC_inl'], open(FILE_gpcr, 'wb'))
pickle.dump(mat['icAdmatDGC_inl'], open(FILE_ic, 'wb'))
pickle.dump(mat['nrAdmatDGC_inl'], open(FILE_nr, 'wb'))

print()
print()

print(np.asarray(mat['DTIMatrix']).shape)
print(np.unique(mat['DTIMatrix']))