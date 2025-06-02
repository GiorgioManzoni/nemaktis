# test that I sent to Guilhem to show that if you return scalars to 
# the functions that define the director field, you get error.
#however it works if you return np.ones_like(x) or np.zeros_like(x)

import numpy as np
import nemaktis as nm 

nfield = nm.DirectorField(
    mesh_lengths=(10,10,10),mesh_dimensions=(40,40,10))

def nx(x,y,z):
    return 0
def ny(x,y,z):
    return 0
def nz(x,y,z):
    return 1

nfield.init_from_funcs(nx,ny,nz)

