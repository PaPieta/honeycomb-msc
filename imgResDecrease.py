# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:20:23 2022

@author: pawel
"""

import numpy as np 
import matplotlib.pyplot as plt
import skimage.io
import scipy.ndimage as scp
import tifffile


I = skimage.io.imread('data/29-2016_29-2016-60kV-zoom-center_recon.tif')
I = I[400:460,:,:]

I_resized = scp.zoom(I,(0.5,0.5,0.5))
tifffile.imsave('data/29-2016_29-2016-60kV-resized.tif',I_resized)