# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 16:59:33 2025

@author: manzoni
"""

import sys
import numpy as np 
import nemaktis as nm 
from scipy.io import loadmat


# set dimensions of director field
nfield = nm.DirectorField(
    mesh_lengths=(800,800,100), # (Lx, Ly, Lz)
    mesh_dimensions=(40,40,10)) # (Nx, Ny, Nz) # fix dimensions

print(np.shape(nfield.vals))

# Load the .mat file (Pouya's director matlab file)
#data = loadmat('C:/Users/manzoni/Desktop/NEMAKTIS/uniform_alignment_matrices.mat')
data = loadmat('C:/Users/manzoni/Desktop/NEMAKTIS/90twist_alignment_matrices.mat')

# shape from matlab after a meshgrid with xy indexing (default)
# (Ny,Nx,Nz) ...yes the y come first
u_matlab = data['u'] 
v_matlab = data['v']
w_matlab = data['w']

# convert to the shape you would have after a numpy meshgrid with ij indexing
u_np = np.transpose(u_matlab, (2, 0, 1))  # New shape [Nz, Ny, Nx]
v_np = np.transpose(v_matlab, (2, 0, 1))
w_np = np.transpose(w_matlab, (2, 0, 1))


# formatting director field from matplotlib arrays into
# the shape (Nz,Ny,Nx,3)
# but here we are not giving the Nx, Ny,Nz but the mashed version
#director = np.concatenate((np.expand_dims(data['w'], axis=3),
#                            np.expand_dims(data['v'], axis=3),
#                            np.expand_dims(data['u'], axis=3)), 3)

director = np.concatenate((np.expand_dims(u_np, axis=3),
                            np.expand_dims(v_np, axis=3),
                            np.expand_dims(w_np, axis=3)), 3)



#D = (np.expand_dims(data['w'], axis=3),
#     np.expand_dims(data['v'], axis=3),
#     np.expand_dims(data['u'], axis=3)
#print("D shape",np.shape(D))

#director = np.concatenate(D,3)

#temp_dir = director.T
#director = temp_dir
#number = 3
#director = np.concatenate((np.expand_dims(director,axis=1),number))

print("director shape",np.shape(director))


#sys.exit()

# set the director to the external arrays after formatting in 
# the shape (Nz,Ny,Nx,3)
nfield.vals = director

nfield.normalize()

# save director field for convenience
# nfield.save_to_vti('C:/Users/manzoni/Desktop/NEMAKTIS/pouya_uniform_director.vti')
#nfield.save_to_vti('C:/Users/manzoni/Desktop/NEMAKTIS/pouya_90twist_director.vti')

mat = nm.LCMaterial(
    lc_field=nfield,ne=1.750,no=1.526,nhost=1.0003,nin=1.51,nout=1.0003)
# add 1 mm-thick glass plate
mat.add_isotropic_layer(nlayer=1.51, thickness=1000)

# create the array of wavelength of the light 
wavelengths = np.linspace(0.4,0.6,20)

# create a light propagator object
sim = nm.LightPropagator(material=mat, 
                         wavelengths=wavelengths, 
                         max_NA_objective=0.4, 
                         max_NA_condenser=0.4, 
                         N_radial_wavevectors=1)

# make the light propagate
output_fields=sim.propagate_fields(method="bpm") 

#save the results of the simulation
# output_fields.save_to_vti("output_uniform_Pouya")
#output_fields.save_to_vti("output_90twist_Pouya")

viewer = nm.FieldViewer(output_fields)
viewer.plot()


