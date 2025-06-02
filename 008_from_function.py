# -*- coding: utf-8 -*-
import sys
import numpy as np 
import nemaktis as nm 
print("Nemaktis properly imported")

# set the dimensions of the director field
nfield = nm.DirectorField(
    mesh_lengths=(50,50,2),mesh_dimensions=(80,80,10))


#create the functions to be used for the director field
q = 2*np.pi/20
def nx(x,y,z):
    return np.cos(q*x)
def ny(x,y,z):
    return np.sin(q*x)
def nz(x,y,z):
    return np.zeros_like(x)

#def nx(x,y,z):
#    temp = np.zeros_like(x)
#    for i in range(len(temp)):
#        temp[i]=5
#    return temp
#def ny(x,y,z):
#    temp = np.zeros_like(x)
#    for i in range(len(temp)):
#        temp[i]=5
#    return temp
#def nz(x,y,z):
#    return np.zeros_like(x)



# initialize the director field with the functions nx, ny, nz
nfield.init_from_funcs(nx,ny,nz)

nfield.normalize()

# save the director in vti format (optional)
nfield.save_to_vti('C:/Users/manzoni/Desktop/NEMAKTIS/MY_SINCOS_DIR.vti')

#create the set up for the liquid cristal 
#mat = nm.LCMaterial(
#    lc_field=nfield,ne=1.750,no=1.526,nhost=1.0003,nin=1.51,nout=1.0003)
mat = nm.LCMaterial(
    lc_field=nfield,
    ne="1.6933+0.0078*lambda^(-2)+0.0028*lambda^(-4)",
    no="1.4990+0.0072*lambda^(-2)+0.0003*lambda^(-4)",
    nhost=1.0003,nin=1.51,nout=1.0003)
# add 1 mm-thick glass plate
mat.add_isotropic_layer(nlayer=1.51, thickness=1000)





# create the array of wavelength of the light 
wavelengths = np.linspace(0.45,0.65,20)

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
output_fields.save_to_vti("OUTPUTFIELD_MYSINCOS")

# Use Nemaktis viewer to see the output
viewer = nm.FieldViewer(output_fields)
viewer.plot()