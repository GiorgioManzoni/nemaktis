# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 17:00:44 2025

@author: manzoni
"""

import sys
import numpy as np 
import nemaktis as nm 
from scipy.io import loadmat
from PIL import Image
from pathlib import Path
import os

#path_directors = 'C:/Users/manzoni/Desktop/SKL/NEMAKTIS/POUYA/result/Director_vectors/'
path_directors = 'C:/Users/manzoni/Desktop/SKL/NEMAKTIS/POUYA/director_clean/'


#name_director =  "QTensor_director_solution-K11_1.01e-11-K22_6.5e-12-K33_1.7e-11-K24_4e-12.mat"

name_director = "QTensor_director_solution-K11_1.1e-11-K22_7.5e-12-K33_1.7e-11-K24_4e-12.mat" 

data = loadmat(path_directors+name_director)