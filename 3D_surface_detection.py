# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:57:45 2022

@author: pawel
"""


import numpy as np 
import matplotlib.pyplot as plt
import skimage.io 
import scipy.ndimage as scp
import slgbuilder

import honeycombUnfold


if __name__ == "__main__":

    I = skimage.io.imread('data/29-2016_29-2016-60kV-zoom-center_recon.tif')
    # I = I[200:800,:,:]
    I = I[400:450,:,:]


    # Median filter for removing salt and pepper-like noise
    I_filt = np.zeros_like(I)
    for i in range(I.shape[0]):
        I_filt[i,:,:] = scp.median_filter(I[i,:,:], size=[5,7])

    I_2d = I_filt[20,:,:]

    plt.figure()
    plt.imshow(I_2d, cmap='gray')
    plt.show()