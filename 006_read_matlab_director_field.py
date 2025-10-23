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
nfield = nm.DirectorField(
    mesh_lengths=(8000,8000,1000), # (Lx, Ly, Lz)
    mesh_dimensions=(40,40,10)) # (Nx, Ny, Nz) # fix dimensions 40 40 10 

# read (u,v,z) director field results of a meshgrid with xy indexing (default)
data = loadmat('C:/Users/manzoni/Desktop/NEMAKTIS/90twist_alignment_matrices.mat')
# (Ny,Nx,Nz) Note! the y axis come first! (although usually Nx=Ny)
u_matlab = data['u'] 
v_matlab = data['v']
w_matlab = data['w']

# convert to the shape you would have after a numpy meshgrid with ij indexing
# as required by Nemaktis
# New shape [Nz, Ny, Nx]
u_np = np.transpose(u_matlab, (2, 0, 1))  
v_np = np.transpose(v_matlab, (2, 0, 1))
w_np = np.transpose(w_matlab, (2, 0, 1))

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
nfield.save_to_vti('C:/Users/manzoni/Desktop/NEMAKTIS/POUYA_90twist_dir.vti')

#create the set up for the liquid cristal 
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

#print(sim.material)
#sys.exit()

# make the light propagate
output_fields=sim.propagate_fields(method="bpm") 

#save the results of the simulation
output_fields.save_to_vti("OUTPUTFIELD_90twist_POUYA")

# Use Nemaktis viewer to see the output
viewer = nm.FieldViewer(output_fields)
viewer.plot()