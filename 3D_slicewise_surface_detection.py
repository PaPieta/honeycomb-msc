# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:57:45 2022

@author: pawel
"""


from os import truncate
import numpy as np 
import matplotlib.pyplot as plt
import skimage.io 
import scipy.ndimage as scp
import slgbuilder
import tifffile

from honeycombUnfoldSlicewise3d import HoneycombUnfoldSlicewise3d
import helpers
import vtk_write_lite as vwl # for saving vtk file

import time


def unfoldedHoneycombSurfDetect(unfolded_img, honeycombCost,  visualize=False, return_helper_surfaces=False):
    """ Detects edges of honeycomb structure based on an unfolded image.\n
    Params:\n
    unfolded_img - clean unfolded image
    honeycombCost - cost function image for the image \n
    visualize - if True, plots the resulting surfaces on an unfolded image \n
    return_helper_surfaces - if True, returns also dark helper surfaces.
    """
    # Layered surface detection parameters - hidden, they don't need to be changed
    # darkSurfacesWeight = 300 # weight of dark surfaces cost 
    # smoothness = [2,1] # honeycomb edge and dark surface smoothness term 
    # honeycombSurfacesMargin = [4, 25] # min, max distance margin between the honeycomb edges
    # darkSurfacesMargin = [10, 35] # min, max distance margin between the dark helper surfaces
    # darkToHoneycombMinMargin = 1 # min distance between the helper surface and honeycomb edge
    darkSurfacesWeight = 300 # weight of dark surfaces cost 
    smoothness = [1,1] # honeycomb edge and dark surface smoothness term 
    honeycombSurfacesMargin = [2, 14] # min, max distance margin between the honeycomb edges
    darkSurfacesMargin = [5, 17] # min, max distance margin between the dark helper surfaces
    darkToHoneycombMinMargin = 0 # min distance between the helper surface and honeycomb edge

    layers = [slgbuilder.GraphObject(0*honeycombCost), slgbuilder.GraphObject(0*honeycombCost), # no on-surface cost
            slgbuilder.GraphObject(darkSurfacesWeight*(255-honeycombCost)), slgbuilder.GraphObject(darkSurfacesWeight*(255-honeycombCost))] # extra 2 dark lines
    helper = slgbuilder.MaxflowBuilder()
    helper.add_objects(layers)

    ## Adding regional costs, 
    # The region in the middle is bright compared to two darker regions.
    helper.add_layered_region_cost(layers[0], 255-honeycombCost, honeycombCost)
    helper.add_layered_region_cost(layers[1], honeycombCost, 255-honeycombCost)

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

def detect3dSlicewiseLayers(img, layer_num, params):
    """ Calls unfolding and layered surface detection methods to 
    detect multiple layers in a 3D honeycomb image using slicewise approach.\n
    Parameters:\n
    img - 2D honecomb image\n
    layer_num - number of layers to detect\n
    params - list of detection parameters:\n
        visualizeUnfolding - if True, visualizes the unfolding process steps\n
        interpStep - distance between the interpolated points\n
        normalLinesRange - Range (half of the length) of lines normal to interpolation points\n
        interpolateFoldedSurface - if True - interpolates unfolded surface to have a value for each x axis pixel\n
        returnHelperSurfaces - if True, returns also dark helper surfaces from surface detection process.
    """

    if len(params)!= 5:
            raise Exception(f'Expected 5 parameters:\
            visualizeUnfolding, interpStep, normalLinesRange,,\
            interpolateFoldedSurface, returnHelperSurfaces, but got only {len(params)}')

    #Extracting parameters from params list
    visualizeUnfolding = params[0]
    interpStep = params[1]
    normalLinesRange = params[2]
    interpolateFoldedSurface = params[3]
    returnHelperSurfaces = params[4]

    ## Calculate image gaussian model 
    means, variances = helpers.imgGaussianModel(img)

    vis_img = np.copy(img)
    vis_img = helpers.rescaleImage(img, 1, 255)
    
    ### Unfolding
    hcList = [HoneycombUnfoldSlicewise3d(img, vis_img, visualize=visualizeUnfolding) for i in range(layer_num)]
    # WARNING!! removed temporarily NOT
    for hc in hcList:
        vis_img = hc.draw_corners() 

    layersList = []
    for i in range(layer_num):
        hc = hcList[i]
        print(f"Layer {i+1} started")
        hc.interpolate_points(step=interpStep)
        print("Points interpolated")
        hc.smooth_interp_corners()
        print("Points smoothed")
        hc.calculate_normals(normals_range=normalLinesRange)
        print("Normals calculated")
        t0 = time.time()
        hc.get_unfold_points_from_normals(interp_points=normalLinesRange*2)
        t1 = time.time()
        print(f"Normals interpolated, time: {t1-t0}")
        unfolded_stack = hc.unfold_image()
        print("Image unfolded")
        
        zstackList = []
        for j in range(unfolded_stack.shape[0]):
            unfolded_img = unfolded_stack[j,:,:]
            # Calculate cost
            honeycombCost = (1 - helpers.sigmoidProbFunction(unfolded_img,means,variances, visualize=(visualizeUnfolding and j == 0)))*255

            if visualizeUnfolding == True and j == 0:
                plt.figure()
                plt.imshow(honeycombCost)
                plt.show()

            unfolded_img = helpers.rescaleImage(unfolded_img, 1, 255)
            ### Layered surfaces detection
            segmentation_surfaces = unfoldedHoneycombSurfDetect(unfolded_img, honeycombCost, 
                visualize=(visualizeUnfolding and j == 0), return_helper_surfaces=returnHelperSurfaces)

            ## Fold detected lines back to original image shape
            folded_surfaces = hc.fold_surfaces_back(segmentation_surfaces, zIdx=j ,interpolate=interpolateFoldedSurface)
            zstackList.append(folded_surfaces)
            print(f"Layer {i+1}, Zstack {j+1}")
        layersList.append(zstackList)
        print(f"Finished Layer {i+1}")
    return layersList


if __name__ == "__main__":

    # I = skimage.io.imread('data/29-2016_29-2016-60kV-resized_z200.tif')
    I = skimage.io.imread('data/29-2016_29-2016-60kV-resized.tif')

    visualizeUnfolding = False # if True - visualizes the unfolding process steps
    interpStep = 10/4 # Distance between the interpolated points
    normalLinesRange = 25 # Range (half of the length) of lines normal to interpolation points
    interpolateFoldedSurface = False  # if True - interpolates unfolded surface to have a value for each x axis pixel
    returnHelperSurfaces = False # if True, returns also dark helper surfaces from surface detection process
    params = [visualizeUnfolding, interpStep, normalLinesRange, 
        interpolateFoldedSurface, returnHelperSurfaces]

    layersList = detect3dSlicewiseLayers(I, 4, params)
    
    # plt.figure()
    # plt.imshow(I[15,:,:], cmap='gray')
    # for i in range(len(layersList)):
    #     folded_surfaces = layersList[i][15]
    #     for j in range(len(folded_surfaces)):
    #         folded_surface = folded_surfaces[j]
    #         if j < 2:
    #             plt.plot(folded_surface[0,:],folded_surface[1,:], 'r')
    #         else:
    #             plt.plot(folded_surface[0,:],folded_surface[1,:], 'b')

    # plt.show()

    segmImg = helpers.layersToMatrix(layersList, I.shape)

    # plt.imshow(segmImg[15,:,:].transpose())
    # plt.show()

    # surf = helpers.layersToSurface(layersList)
    # surf_array = np.array(surf)
    # np.save("data/slicewise_zXXX_allSurf_raw.npy", surf_array)

    # vwl.save_multSurf2vtk('data/surfaces/slicewise_zXXX_allSurf.vtk', surf)