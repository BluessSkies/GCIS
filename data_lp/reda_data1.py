import numpy as np
import pickle as pkl
import sys
import torch
from scipy import sparse


tx = np.random.randint(0,2,(1000,1433)).astype(np.float32)
x = np.random.randint(0,2,(140,1433)).astype(np.float32)
allx = np.random.randint(0,2,(1708,1433)).astype(np.float32)

stx = sparse.csr_matrix(tx)
sx = sparse.csr_matrix(x)
sallx = sparse.csr_matrix(allx)

file = open('ind.cora.x', 'wb')
pkl.dump(sx, file)
file.close()

file = open('ind.cora.tx', 'wb')
pkl.dump(tx, file)
file.close()

file = open('ind.cora.allx', 'wb')
pkl.dump(allx, file)
file.close()


objects = []
with open("ind.cora.x_old", 'rb') as f:
    if sys.version_info > (3, 0):
        objects.append(pkl.load(f, encoding='latin1'))
    else:
        objects.append(pkl.load(f))
xx = objects[0].toarray()