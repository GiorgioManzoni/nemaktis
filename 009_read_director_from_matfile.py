# -*- coding: utf-8 -*-
"""
Created on Fri May  2 17:03:58 2025

@author: manzoni
"""

import sys
import numpy as np 
import nemaktis as nm 
from scipy.io import loadmat

# set dimensions of director field
#nfield = nm.DirectorField(
#    mesh_lengths=(20,20,5), # (Lx, Ly, Lz)
#    mesh_dimensions=(40,40,10)) # (Nx, Ny, Nz) # fix dimensions 40 40 10 

#nfield = nm.DirectorField(
#    mesh_lengths=(20,20,5), # (Lx, Ly, Lz)
#    mesh_dimensions=(100,100,40)) # (Nx, Ny, Nz) # fix dimensions 40 40 10 

nfield = nm.DirectorField(
    mesh_lengths=(100,100,2), # (Lx, Ly, Lz)
    mesh_dimensions=(80,80,10)) # (Nx, Ny, Nz) # fix dimensions 40 40 10 


# read (u,v,z) director field results of a meshgrid with xy indexing (default)
#data = loadmat('C:/Users/manzoni/Desktop/SKL/NEMAKTIS/6_QTensor_director_solution.mat')
#data = loadmat('C:/Users/manzoni/Desktop/SKL/NEMAKTIS/POUYA/QTensor_director_solution_K11_0-6.mat')
#data = loadmat('C:/Users/manzoni/Desktop/SKL/NEMAKTIS/POUYA/QTensor_director_solution_K11_1-1.mat')
data = loadmat('C:/Users/manzoni/Desktop/SKL/NEMAKTIS/POUYA/QTensor_director_solution_K11_1-6.mat')

# (Ny,Nx,Nz) Note! the y axis come first! (although usually Nx=Ny)
u_matlab = data['u_final']  # (3,80,10)
v_matlab = data['v_final']
w_matlab = data['w_final']

#THE FOLLOWING COMMAND IS TO TAKE THE FIRST COLUMN AND EXAPAND IT FOR 80 TIMES 
# tO HAVE THE RIGHT DIMENSIONALITY (AND SAVE COMPUTER POWER)
# Take first row and expand dimensions, then repeat
u_new = np.tile(u_matlab[0][np.newaxis, :, :], (80, 1, 1)) #(80,80,10)
v_new = np.tile(v_matlab[0][np.newaxis, :, :], (80, 1, 1))
w_new = np.tile(w_matlab[0][np.newaxis, :, :], (80, 1, 1))

print(u_matlab.shape)
print(u_new.shape)

#sys.exit()

# convert to the shape you would have after a numpy meshgrid with ij indexing
# as required by Nemaktis
# New shape [Nz, Ny, Nx]
#u_np = np.transpose(u_matlab, (2, 0, 1))  
#v_np = np.transpose(v_matlab, (2, 0, 1))
#w_np = np.transpose(w_matlab, (2, 0, 1))

u_np = np.transpose(u_new, (2, 0, 1))  
v_np = np.transpose(v_new, (2, 0, 1))
w_np = np.transpose(w_new, (2, 0, 1))


# putting the three component of the director in a single variable
# Nemaktis format (Nz,Ny,Nx,3)
director = np.concatenate((np.expand_dims(u_np, axis=3),
                            np.expand_dims(v_np, axis=3),
                            np.expand_dims(w_np, axis=3)), 3)

# set the nemaktis object for the director field to 
# the value for the director field we have just formatted
nfield.vals = director
# normalize it (optional)
nfield.normalize()

# save the director in vti format (optional)
nfield.save_to_vti('C:/Users/manzoni/Desktop/NEMAKTIS/PN1dir.vti')

#create the set up for the liquid cristal 
mat = nm.LCMaterial(
    lc_field=nfield,ne=1.750,no=1.526,nhost=1.0003,nin=1.51,nout=1.0003)
# add 1 mm-thick glass plate
mat.add_isotropic_layer(nlayer=1.51, thickness=1000)





# create the array of wavelength of the light 
wavelengths = np.linspace(0.4,0.6,10)

# create a light propagator object
sim = nm.LightPropagator(material=mat, 
                         wavelengths=wavelengths, 
                         max_NA_objective=0.4, 
                         max_NA_condenser=0.4, 
                         N_radial_wavevectors=1)

#print(sim.material)
#sys.exit()

# make the light propagate
output_fields=sim.propagate_fields(method="bpm") 

#save the results of the simulation
output_fields.save_to_vti("PN1output.vti")

# Use Nemaktis viewer to see the output
viewer = nm.FieldViewer(output_fields)
viewer.plot()