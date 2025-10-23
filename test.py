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

