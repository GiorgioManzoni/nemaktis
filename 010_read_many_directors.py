# -*- coding: utf-8 -*-
"""
Created on Fri May  2 17:03:58 2025

@author: manzoni
"""

import sys
import numpy as np 
import nemaktis as nm 
from scipy.io import loadmat
from PIL import Image
from pathlib import Path
import os

def get_img_np(viewer,polariser_angle=0,analyser_angle=90,grayscale=False):
    viewer.analyser_angle = analyser_angle;
    viewer.polariser_angle = polariser_angle;
    viewer.grayscale = grayscale;
    viewer.update_image()
    img = viewer.get_image()
    return img

def save_np_img(img,path,img_name):
    # If last dim is 1 (grayscale), remove the singleton dimension
    if img.shape[2] == 1:
        img_to_save = img[:, :, 0]
    else:
        img_to_save = img
    # Convert to uint8 if necessary (common image dtype)
    if img_to_save.dtype != np.uint8:
        img_to_save = (img_to_save * 255).astype(np.uint8)  # if img is float in [0,1]
    # Create PIL image
    if img_to_save.ndim == 2:  # grayscale
        pil_img = Image.fromarray(img_to_save, mode='L')
    else:  # color
        pil_img = Image.fromarray(img_to_save, mode='RGB')
    # Define path
    output_path = Path(path) / img_name
    # Create folder if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save image
    pil_img.save(output_path)

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


#print(vars(nfield))
#print(nfield._Nx,nfield._Ny,nfield._Nz)
#nfield._Nx = 80
#print(nfield._Nx,nfield._Ny,nfield._Nz)
#sys.exit()

#path_directors = 'C:/Users/manzoni/Desktop/SKL/NEMAKTIS/POUYA/result/Director_vectors/'
path_directors = 'C:/Users/manzoni/Desktop/SKL/NEMAKTIS/POUYA/director_clean/'
#name_director = 'QTensor_director_solution-K11_1.01e-11-K22_6.5e-12-K33_1.7e-11-K24_4e-12.mat'

#print(name_director[:-3]+'png')
i = 0
for root, dirs, files in os.walk(path_directors):
    for file_name in files:
        full_path = os.path.join(root, file_name)
        #print(full_path)
        if ('.mat' in file_name) and ('Sin' not in file_name):            
            try:
                i = i +1
                #print("######")
                #print(i, ' ', file_name)
                #print("######")
                
                # read (u,v,z) director field results of a meshgrid with xy indexing (default)
                #data = loadmat('C:/Users/manzoni/Desktop/SKL/NEMAKTIS/6_QTensor_director_solution.mat')
                #data = loadmat('C:/Users/manzoni/Desktop/SKL/NEMAKTIS/POUYA/QTensor_director_solution_K11_0-6.mat')
                #data = loadmat('C:/Users/manzoni/Desktop/SKL/NEMAKTIS/POUYA/QTensor_director_solution_K11_1-1.mat')
                #data = loadmat('C:/Users/manzoni/Desktop/SKL/NEMAKTIS/POUYA/QTensor_director_solution_K11_1-6.mat')
                
                name_director = file_name
                
                # Check if file exists and has content
                #file_path = path_directors+name_director
                #print(f"File exists: {os.path.exists(file_path)}")
                #print(f"File size: {os.path.getsize(file_path)} bytes")
                
                #sys.exit()
                #try:
                #    data = loadmat(path_directors+name_director)
                #    print(name_director)
                #except:
                #    continue
                #    print(name_director)
                #continue
                
                data = loadmat(path_directors+name_director)
            
                # (Ny,Nx,Nz) Note! the y axis come first! (although usually Nx=Ny)
                u_matlab = data['u_final']  # (3,80,10)
                v_matlab = data['v_final']
                w_matlab = data['w_final']
                
                print(v_matlab.shape)
                print(v_matlab.shape[0])  
                print(v_matlab.shape[1])  
                print(v_matlab.shape[2])  
                
                Nx_matlab = v_matlab.shape[0]
                Ny_matlab = v_matlab.shape[1]
                Nz_matlab = v_matlab.shape[2]
                
                
                if Nx_matlab == 3:
                    # it means we simulated only three vectors as it is constant so 
                    # we take the constant value and duplicate to the dimension of Ny
                    # Take first row and expand dimensions, then repeat
                    u_new = np.tile(u_matlab[0][np.newaxis, :, :], (Ny_matlab, 1, 1)) #(3,80,10)-->(80,80,10)
                    v_new = np.tile(v_matlab[0][np.newaxis, :, :], (Ny_matlab, 1, 1))
                    w_new = np.tile(w_matlab[0][np.newaxis, :, :], (Ny_matlab, 1, 1))                    
                    Nx_final = Ny_matlab  
                else:
                    u_new = u_matlab
                    v_new = v_matlab
                    w_new = w_matlab
                    Nx_final = Nx_matlab

                Ny_final = Ny_matlab
                Nz_final = Nz_matlab 
                
                print(u_matlab.shape)
                print(u_new.shape)
                print(Nx_final,Ny_final,Nz_final)                
                
                nfield = nm.DirectorField(mesh_lengths=(100,100,2), # (Lx, Ly, Lz) microns
                                          mesh_dimensions=(Nx_final,Ny_final,Nz_final))
                
            
                
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
                #nfield.save_to_vti('C:/Users/manzoni/Desktop/NEMAKTIS/PN1dir.vti')
                nfield.save_to_vti(path_directors+name_director[:-3]+'_dir.vti')
             
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
                #output_fields.save_to_vti("PN1output.vti")
                output_fields.save_to_vti(path_directors+name_director[:-3]+'_out_field.vti')
                
                # Use Nemaktis viewer to see the output
                viewer = nm.FieldViewer(output_fields)
                
                #viewer.plot()
                img = viewer.get_image()
                
                img = get_img_np(viewer,polariser_angle=0,analyser_angle=90,grayscale=False)
                
                save_np_img(img,path=path_directors ,img_name=name_director[:-3]+'png')
            except:
                print('######################################################################')
                print('######################################################################')
                print('ERROR:')
                print(i, " ", path_directors+name_director)
                print('######################################################################')
                print('######################################################################')       
                
