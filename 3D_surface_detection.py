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

    I = skimage.io.imread('data/29-2016_29-2016-60kV-resized.tif')
    # I = I[200:800,:,:]
    # I = I[400:430,:,:]


    # # Median filter for removing salt and pepper-like noise
    # I_filt = np.zeros_like(I)
    # for i in range(I.shape[0]):
    #     I_filt[i,:,:] = scp.median_filter(I[i,:,:], size=[5,7])

    vis_img= np.copy(I)

    hc = honeycombUnfold.HoneycombUnfold3d(I, vis_img, visualize=True)
    vis_img = hc.draw_corners() 
    hc.interpolate_points(step=2)
    hc.smooth_interp_corners()
    hc.calculate_normals()

    # I_2d = I_filt[15,:,:]

    # plt.figure()
    # plt.imshow(I_2d, cmap='gray')
    # plt.show()