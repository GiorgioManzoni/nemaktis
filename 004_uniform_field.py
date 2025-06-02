# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:14:54 2025
@author: manzoni
"""
import sys
import numpy as np 
import nemaktis as nm 

# set dimensions of director field
nfield = nm.DirectorField(
    mesh_lengths=(10,10,10),mesh_dimensions=(40,40,10))

# define functions for uniform director field
def nx(x,y,z):
    return np.zeros_like(x)
def ny(x,y,z):
    return np.zeros_like(x)
def nz(x,y,z):
    return np.ones_like(x)

# initialize the director field 
nfield.init_from_funcs(nx,ny,nz)

nfield.normalize()

## check sizes
#print(nfield.vals)
#print(np.shape(nfield.vals))
#print(type(nfield.vals))

nfield.save_to_vti('C:/Users/manzoni/Desktop/NEMAKTIS/my_unif_xyz_010.vti')

mat = nm.LCMaterial(
    lc_field=nfield,ne=1.750,no=1.526,nhost=1.0003,nin=1.51,nout=1.0003)

# add 1 mm-thick glass plate
# mat.add_isotropic_layer(nlayer=1.51, thickness=1000)

# add empty space of 5 microns
mat.add_isotropic_layer(nlayer=1.0003, thickness=5)

# add 1 mm-thick glass plate
mat.add_isotropic_layer(nlayer=1.51, thickness=1000)

# create the array of wavelength of the light 
wavelengths = np.linspace(0.4,0.6,11)

# create a light propagator object
sim = nm.LightPropagator(material=mat, 
                         wavelengths=wavelengths, 
                         max_NA_objective=0.4, 
                         max_NA_condenser=0, 
                         N_radial_wavevectors=1)
# make the light propagate
output_fields=sim.propagate_fields(method="bpm") 

#save the results of the simulation
output_fields.save_to_vti("output_uniform_xyz_010")

viewer = nm.FieldViewer(output_fields)
viewer.plot()
