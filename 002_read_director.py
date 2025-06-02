# -*- coding: utf-8 -*-
import sys
import numpy as np 
import nemaktis as nm 
print("Nemaktis properly imported")

# set the dimensions of the director field
nfield = nm.DirectorField(
    mesh_lengths=(10,10,10),mesh_dimensions=(40,40,10))

from scipy.io import loadmat
# Load the .mat file
data = loadmat('C:/Users/manzoni/Desktop/NEMAKTIS/uniform_alignment_matrices.mat')

#print(len(data))
#print(len(data['u']))
#sys.exit()

#nx = data['u']
#ny = data['v']
#nz = data['w']

#nx = data['u'][0]
#ny = data['v'][1]
#nz = data['w'][2]

#print(len(nx))
#print(len(ny))
#print(len(nz))


# def nx(x,y,z):
#     return 0
# def ny(x,y,z):
#     return 0
# def nz(x,y,z):
#     return 1

def nx(x,y,z):
    return np.ones_like(x)
def ny(x,y,z):
    return np.ones_like(x)
def nz(x,y,z):
    return np.ones_like(x)


# initialize the director field with the functions nx, ny, nz
nfield.init_from_funcs(nx,ny,nz)



# initialize the director field with the functions nx, ny, nz
#nfield.init_from_funcs(nx,ny,nz)

#nfield.vals = np.array([nz,ny,nx,3])
#print(nfield.vals.shape)
#print(nfield.vals)
#t = np.array((nz,ny,nx,3),dtype=np.ndarray)
#print(t.shape)


#sys.exit()


#normalize the field in case it is not already
nfield.normalize()

# rotate and extend to xy plane
nfield.rotate_90deg("x")
nfield.extend(2, 2)
nfield.set_mask(mask_type="droplet")

# save into vti file
#nfield.save_to_vti("double_twist_droplet")
nfield.save_to_vti('C:/Users/manzoni/Desktop/NEMAKTIS/unif.vti')

# you can also import directly from vti file
# nfield = nm.DirectorField(vti_file="double_twist_droplet.vti")

# Now we can create the liquid crystal material
mat = nm.LCMaterial(
    lc_field=nfield,ne=1.5,no=1.7,nhost=1.55,nin=1.51,nout=1)

# add 5 microns space between the droplet and the glass plate
mat.add_isotropic_layer(nlayer=1.55, thickness=5)
# add 1 mm-thick glass plate
mat.add_isotropic_layer(nlayer=1.51, thickness=1000)

# create the array of wavelength of the light 
wavelengths = np.linspace(0.4,0.8,4) #11)
# create a light propagator object
sim = nm.LightPropagator(material=mat, 
                         wavelengths=wavelengths, 
                         max_NA_objective=0.4, 
                         max_NA_condenser=0, 
                         N_radial_wavevectors=1)
# make the light propagate
output_fields=sim.propagate_fields(method="bpm") 

#save the results of the simulation
output_fields.save_to_vti("output_uniform")
#nfield.save_to_vti("simple_out")

# for reimporting the simualtion from file (you don't need to use it now)
# output_fields = nm.OpticalFields(vti_file="optical_fields.vti")

viewer = nm.FieldViewer(output_fields)
viewer.plot()


####################################
print('Everything seems to work...')

