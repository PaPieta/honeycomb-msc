# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 17:10:52 2022

@author: pawel
"""

import numpy as np 
import matplotlib.pyplot as plt
import skimage.io 
import scipy.ndimage as scp
import slgbuilder

import honeycombUnfold



def rescaleImage(img, minVal, maxVal):
    """Rescales input image to a given range.\n
    Params:\n
    img - image to rescale\n
    minVal - minimum pixel value\n
    maxVal - maximum pixel value.
    """
    img = ((img - img.min()) * (minVal/(img.max() - img.min()) * maxVal)).astype(np.int32)
    return img


def unfoldedHoneycombSurfDetect(unfolded_img, visualize=False, return_helper_surfaces=False):
    """ Detects edges of honeycomb structure based on an unfolded image.\n
    Params:
    unfolded_img - unfolded image of a honeycomb\n
    visualize - if True, plots the resulting surfaces on an unfolded image
    return_helper_surfaces - if True, returns also dark helper surfaces.
    """
    # Layered surface detection parameters - hidden, they don't need to be changed
    darkSurfacesWeight = 300 # weight of dark surfaces cost 
    smoothness = [2,1] # honeycomb edge and dark surface smoothness term 
    honeycombSurfacesMargin = [4, 25] # min, max distance margin between the honeycomb edges
    darkSurfacesMargin = [10, 35] # min, max distance margin between the dark helper surfaces
    darkToHoneycombMinMargin = 1 # min distance between the helper surface and honeycomb edge

    layers = [slgbuilder.GraphObject(0*unfolded_img), slgbuilder.GraphObject(0*unfolded_img), # no on-surface cost
            slgbuilder.GraphObject(darkSurfacesWeight*unfolded_img), slgbuilder.GraphObject(darkSurfacesWeight*unfolded_img)] # extra 2 dark lines
    helper = slgbuilder.MaxflowBuilder()
    helper.add_objects(layers)

    ## Adding regional costs, 
    # The region in the middle is bright compared to two darker regions.
    helper.add_layered_region_cost(layers[0], unfolded_img, np.max(unfolded_img)-unfolded_img)
    helper.add_layered_region_cost(layers[1], np.max(unfolded_img)-unfolded_img, unfolded_img)

    ## Adding geometric constrains
    # Blocks crossing from bottom to top of the image
    helper.add_layered_boundary_cost() 
    # Surface smoothness term
    helper.add_layered_smoothness(layers[0:2],delta=smoothness[0], wrap=False)
    helper.add_layered_smoothness(layers[2:4],delta=smoothness[1], wrap=False)
    # Honeycomb edges pair  
    helper.add_layered_containment(layers[0], layers[1], min_margin=honeycombSurfacesMargin[0], max_margin=honeycombSurfacesMargin[1])
    # Dark helper surfaces 
    helper.add_layered_containment(layers[2], layers[3], min_margin=darkSurfacesMargin[0], max_margin=darkSurfacesMargin[1])
    # Top dark surface and top honeycomb edge 
    helper.add_layered_containment(layers[2], layers[0], min_margin=darkToHoneycombMinMargin) 
    # Bottom honeycomb edge and bottom dark surface
    helper.add_layered_containment(layers[1], layers[3], min_margin=darkToHoneycombMinMargin) 

    ## Cut
    helper.solve()
    segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
    segmentation_lines = [s.shape[0] - np.argmax(s[::-1,:], axis=0) - 1 for s in segmentations]

    ## Unfolded image visualization
    if visualize == True:
        plt.figure()
        plt.imshow(unfolded_img, cmap='gray')
        for i in range(len(segmentation_lines)):
            if i < 2:
                plt.plot(segmentation_lines[i], 'r')
            else:
                plt.plot(segmentation_lines[i], 'b')

        plt.show()

    if return_helper_surfaces == False:
        segmentation_lines.pop(-1)
        segmentation_lines.pop(-1)
    
    return segmentation_lines


def detect2dLayers(img, layer_num, params):
    """ Calls unfolding and layered surface detection methods to 
    detect multiple layers in a 2D honeycomb image.\n
    Parameters:\n
    img - 2D honecomb image\n
    layer_num - number of layers to detect\n
    params - list of detection parameters:\n
        visualizeUnfolding - if True, visualizes the unfolding process steps\n
        interpPointsScale - multiplier of the amount points between the corners that will be interpolated
            1 equals the distance value between the points\n
        normalLinesRange - Range (half of the length) of lines normal to interpolation points\n
        interpolateFoldedSurface - if True - interpolates unfolded surface to have a value for each x axis pixel\n
        returnHelperSurfaces - if True, returns also dark helper surfaces from surface detection process.
    """

    if len(params)!= 5:
            raise Exception(f'Expected 5 parameters:\
            visualizeUnfolding, interpPointsScale, normalLinesRange,,\
            interpolateFoldedSurface, returnHelperSurfaces, but got only {len(params)}')

    #Extracting parameters from params list
    visualizeUnfolding = params[0]
    interpPointsScale = params[1]
    normalLinesRange = params[2]
    interpolateFoldedSurface = params[3]
    returnHelperSurfaces = params[4]

    ### Unfolding
    hcList = []
    vis_img = rescaleImage(img, 1, 255)
    for i in range(layer_num):
        hc = honeycombUnfold.HoneycombUnfold2d(img, vis_img, visualize=visualizeUnfolding)
        vis_img = hc.draw_corners() 
        hcList.append(hc)

    layersList = []
    for i in range(layer_num):
        hc = hcList[i]
        hc.interpolate_points(points_scale=interpPointsScale)
        hc.smooth_interp_corners()
        hc.calculate_normals(normals_range=normalLinesRange)
        hc.get_unfold_points_from_normals(interp_points= normalLinesRange*2)
        unfolded_img = hc.unfold_image()
        unfolded_img = rescaleImage(unfolded_img, 1, 255)

        ### Layered surfaces detection
        segmentation_surfaces = unfoldedHoneycombSurfDetect(unfolded_img, 
            visualize=visualizeUnfolding, return_helper_surfaces=returnHelperSurfaces)

        ## Fold detected lines back to original image shape
        folded_surfaces = hc.fold_surfaces_back(segmentation_surfaces,interpolate=interpolateFoldedSurface)
        layersList.append(folded_surfaces)
    return layersList

if __name__ == "__main__":

    I = skimage.io.imread('data/29-2016_29-2016-60kV-zoom-center_recon.tif')

    I_2d = I[195,:,:]
    # Median filter for removing salt and pepper-like noise
    I_2d_filt = scp.median_filter(I_2d, size=[5,7])

    visualizeUnfolding = False # if True - visualizes the unfolding process steps
    interpPointsScale = 0.4 # -multiplier of the amount points between the corners that will be interpolated 
                        # 1 equals the distance value between the points
    normalLinesRange = 50 # Range (half of the length) of lines normal to interpolation points
    interpolateFoldedSurface = True # if True - interpolates unfolded surface to have a value for each x axis pixel
    returnHelperSurfaces = False # if True, returns also dark helper surfaces from surface detection process
    params = [visualizeUnfolding, interpPointsScale, normalLinesRange, 
        interpolateFoldedSurface, returnHelperSurfaces]

    layersList = detect2dLayers(I_2d_filt, 4, params)

    plt.figure()
    plt.imshow(I_2d_filt, cmap='gray')
    for i in range(len(layersList)):
        folded_surfaces = layersList[i]
        for j in range(len(folded_surfaces)):
            folded_surface = folded_surfaces[j]
            if j < 2:
                plt.plot(folded_surface[0,:],folded_surface[1,:], 'r')
            else:
                plt.plot(folded_surface[0,:],folded_surface[1,:], 'b')

    plt.show()