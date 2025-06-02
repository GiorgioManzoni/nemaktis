# -*- coding: utf-8 -*-
import sys
import numpy as np 
import nemaktis as nm 
print("Nemaktis properly imported")

# set the dimensions of the director field
nfield = nm.DirectorField(
    mesh_lengths=(10,10,10),mesh_dimensions=(80,80,80))


#create the functions to be used for the director field
q = 2*np.pi/20
def nx(x,y,z):
    r = np.sqrt(x**2+y**2)
    return -q*y*np.sinc(q*r)
def ny(x,y,z):
    r = np.sqrt(x**2+y**2)
    return q*x*np.sinc(q*r)
def nz(x,y,z):
    r = np.sqrt(x**2+y**2)
    return np.cos(q*r)


# initialize the director field with the functions nx, ny, nz
nfield.init_from_funcs(nx,ny,nz)

#normalize the field in case it is not already
nfield.normalize()

# rotate and extend to xy plane

print(np.shape(nfield.vals))


nfield.rotate_90deg("x")
#print(help(nfield.rotate_90deg))
#print(np.shape(nfield.vals))
#nfield.extend(2, 2)

print(np.shape(nfield.vals))
print(help(nfield.extend))

nfield.set_mask(mask_type="droplet")

print(np.shape(nfield.vals))


sys.exit()


# save into vti file
nfield.save_to_vti("double_twist_droplet")
#nfield.save_to_vti("simple")

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
output_fields.save_to_vti("optical_fields")
#nfield.save_to_vti("simple_out")

# for reimporting the simualtion from file (you don't need to use it now)
# output_fields = nm.OpticalFields(vti_file="optical_fields.vti")

viewer = nm.FieldViewer(output_fields)
viewer.plot()


####################################
print('Everything seems to work...')

