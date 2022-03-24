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
import copy
from skimage.morphology import closing, dilation

from honeycombUnfoldSlicewise3d import HoneycombUnfoldSlicewise3d
import helpers
import vtk_write_lite as vwl # for saving vtk file

import time


def unfoldedHoneycombHelperDetect(unfolded_img, honeycombCost,visualize=False):
    """Detects helper surface in the center of the honeycomb strucutre. Used for modifying the cost function.
    Params:\n
    unfolded_img - clean unfolded image
    honeycombCost - cost function image for the honeycomb-targeting surfaces \n
    visualize - if True, plots the resulting surfaces on an unfolded image.
    """
    smoothness = 1

    layers = [slgbuilder.GraphObject(honeycombCost)]
    helper = slgbuilder.MaxflowBuilder(flow_type=np.float64)
    helper.add_objects(layers)
    
    helper.add_layered_boundary_cost() 

    helper.add_layered_smoothness(layers,delta=smoothness, wrap=True)

     ## Cut
    helper.solve()
    segmentations = [helper.get_labels(l).astype(np.int32) for l in layers]
    segmentation_lines = [s.shape[0] - np.argmax(s[::-1,:], axis=0) - 1 for s in segmentations]
    #Correct for 0.5 pixel shift
    segmentation_lines = [x+0.5 for x in segmentation_lines]

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
    return segmentation_lines

def unfoldedHoneycombSurfDetect(unfolded_img, honeycombCost, backgroundCost, visualize=False, return_helper_surfaces=False, darkSurfWeight=20):
    """ Detects edges of honeycomb structure based on an unfolded image.\n
    Params:\n
    unfolded_img - clean unfolded image
    honeycombCost - cost function image for the honeycomb-targeting surfaces \n
    backgroundCost - cost function image for the background-targeting surfaces \n
    visualize - if True, plots the resulting surfaces on an unfolded image \n
    return_helper_surfaces - if True, returns also dark helper surfaces.
    """
    # Layered surface detection parameters - hidden, they don't need to be changed
    darkSurfacesWeight = darkSurfWeight # weight of dark surfaces cost 
    smoothness = [1,1] # honeycomb edge and dark surface smoothness term 
    honeycombSurfacesMargin = [4, 16] # min, max distance margin between the honeycomb edges
    darkSurfacesMargin = [6, 20] # min, max distance margin between the dark helper surfaces
    darkToHoneycombMinMargin = 1 # min distance between the helper surface and honeycomb edge
    darkWhiteHelperSurfacesMargin = [1,10] # min, max distance margin between the dark helper surface and white helper surface
    # darkSurfacesWeight = 300 # weight of dark surfaces cost 
    # smoothness = [1,1] # honeycomb edge and dark surface smoothness term 
    # honeycombSurfacesMargin = [2, 14] # min, max distance margin between the honeycomb edges
    # darkSurfacesMargin = [5, 17] # min, max distance margin between the dark helper surfaces
    # darkToHoneycombMinMargin = 0 # min distance between the helper surface and honeycomb edge

    layers = [slgbuilder.GraphObject(0*honeycombCost), slgbuilder.GraphObject(0*honeycombCost), # no on-surface cost
            slgbuilder.GraphObject(darkSurfacesWeight*backgroundCost), slgbuilder.GraphObject(darkSurfacesWeight*backgroundCost), # extra 2 dark lines
            slgbuilder.GraphObject(darkSurfacesWeight*1.2*(honeycombCost))] # Extra white line 
    helper = slgbuilder.MaxflowBuilder(flow_type=np.float64)
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
    # helper.add_layered_smoothness(layers[2:4],delta=smoothness[1], wrap=False)
    helper.add_layered_smoothness(layers[2:5],delta=smoothness[1], wrap=True)
    # Honeycomb edges pair  
    helper.add_layered_containment(layers[0], layers[1], min_margin=honeycombSurfacesMargin[0], max_margin=honeycombSurfacesMargin[1])
    # Dark helper surfaces 
    helper.add_layered_containment(layers[2], layers[3], min_margin=darkSurfacesMargin[0], max_margin=darkSurfacesMargin[1])
    # Top dark surface and top honeycomb edge 
    helper.add_layered_containment(layers[2], layers[0], min_margin=darkToHoneycombMinMargin) 
    # Bottom honeycomb edge and bottom dark surface
    helper.add_layered_containment(layers[1], layers[3], min_margin=darkToHoneycombMinMargin) 

    # Top dark surface and white central surface
    helper.add_layered_containment(layers[2], layers[4], min_margin=darkWhiteHelperSurfacesMargin[0], max_margin=darkWhiteHelperSurfacesMargin[1]) 
    # White central surface and bottom dark surface
    helper.add_layered_containment(layers[4], layers[3], min_margin=darkWhiteHelperSurfacesMargin[0], max_margin=darkWhiteHelperSurfacesMargin[1]) 

    ## Cut
    helper.solve()
    segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
    segmentation_lines = [s.shape[0] - np.argmax(s[::-1,:], axis=0) - 1 for s in segmentations]
    #Correct for 0.5 pixel shift
    segmentation_lines = [x+0.5 for x in segmentation_lines]

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
        segmentation_lines.pop(-1)
    
    return segmentation_lines

def detect3dSlicewiseLayers(img, layer_num, params, savePointsPath='', loadPointsPath=''):
    """ Calls unfolding and layered surface detection methods to 
    detect multiple layers in a 3D honeycomb image using slicewise approach.\n
    Parameters:\n
    img - 2D honecomb image\n
    layer_num - number of layers to detect\n
    params - list of detection parameters:\n
        visualizeUnfolding - if True, visualizes the unfolding process steps\n
        interpStep - distance between the interpolated points\n
        normalLinesRange - Range (half of the length) of lines normal to interpolation points\n
        normalLinesNumPoints - Number of interpolation points along a normal line\n
        returnHelperSurfaces - if True, returns also dark helper surfaces from surface detection process.\n
    savePointsPath - if not '', saves manually marked points as a np array
    loadPointsPath - if not '', loads the points from given path instead of initializing manual marking.
    """

    if len(params)!= 5:
            raise Exception(f'Expected 5 parameters: visualizeUnfolding, interpStep, normalLinesRange, returnHelperSurfaces, but got {len(params)}')

    #Extracting parameters from params list
    visualizeUnfolding = params[0]
    interpStep = params[1]
    normalLinesRange = params[2]
    normalLinesNumPoints = params[3]
    returnHelperSurfaces = params[4]

    ## Calculate image gaussian model 
    means, variances = helpers.imgGaussianModel(img)

    # Prepare parabola vector
    parVec = np.arange(-normalLinesNumPoints/2,normalLinesNumPoints/2,1)
    parVec = 0.05*parVec**2+1

    vis_img = np.copy(img)
    vis_img = helpers.rescaleImage(img, 1, 255)

    helperStack = np.zeros(img.shape)
    
    ### Unfolding
    hcList = [HoneycombUnfoldSlicewise3d(img, vis_img, visualize=visualizeUnfolding) for i in range(layer_num)]
    if loadPointsPath == '':
        for hc in hcList:
            vis_img = hc.draw_corners()
    else:
        hcList = helpers.loadHcPoints(hcList, loadPointsPath)

    # SaVing the points
    if savePointsPath != '':
        helpers.saveHcPoints(hcList,savePointsPath)
         
    # First loop to roughly detect the centers (a helper detection)
    for i in range(layer_num):
        hc = hcList[i]
        print(f"Helper detection, Layer {i+1} started")
        hc.interpolate_points(step=interpStep)
        print("Points interpolated")
        hc.smooth_interp_corners()
        print("Points smoothed")
        t0 = time.time()
        hc.calculate_normals(normals_range=normalLinesRange)
        t1 = time.time()
        print(f"Normals calculated, time: {t1-t0}")
        t0 = time.time()
        hc.get_unfold_points_from_normals3(interp_points=normalLinesNumPoints)
        t1 = time.time()
        print(f"Normals interpolated, time: {t1-t0}")
        unfolded_stack = hc.unfold_image()
        print("Image unfolded")

        # Calculate cost
        helperCost_stack = (1 - helpers.sigmoidProbFunction(unfolded_stack,means,variances, weight=0.001, visualize=visualizeUnfolding))*255
        helperCost_stack = scp.uniform_filter(helperCost_stack,size=3)
        # Add the parabola
        helperCost_stack = np.moveaxis((np.moveaxis(helperCost_stack,1,-1)+parVec),-1,1)
        
        for j in range(unfolded_stack.shape[0]):
            unfolded_img = unfolded_stack[j,:,:]
            helperCost = helperCost_stack[j,:,:]

            unfolded_img = helpers.rescaleImage(unfolded_img, 1, 255)
            #calculate helper line
            helperSurf = unfoldedHoneycombHelperDetect(unfolded_img, helperCost, visualize=visualizeUnfolding)

            ## Fold detected lines back to original image shape
            folded_helper = np.round(hc.fold_surfaces_back(helperSurf, zIdx=j)[0]).astype('int')
            #Apply detection to a helper stack
            helperStack[j,folded_helper[1,:],folded_helper[0,:]]=i+1
            print(f"Helper layer {i+1}, Zstack {j+1}")
        print(f"Finished layer {i+1} for the helper detection")

    # Make a copy of the hc objects with different image
    hcHelpList = copy.deepcopy(hcList)
    for i in range(layer_num):
        hcHelp = hcHelpList[i]
        hcHelp.img = helperStack


    # Main detection loop
    layersList = []
    for i in range(layer_num):
        hc = hcList[i]
        hcHelp = hcHelpList[i]
        print(f"Final detection, layer {i+1} started")
        unfolded_stack = hc.unfold_image()
        unfolded_helperStack = hcHelp.unfold_image(method='nearest')
        print("Image unfolded")

        # Fix the helper to be consistent
        meanLayerHeight = np.mean(np.where(unfolded_helperStack==i+1)[1])
        unfolded_helperStackFixed = np.zeros(unfolded_helperStack.shape)
        unique_vals = np.unique(unfolded_helperStack)
        # Loop through masks of the honeycomb layers
        for j in range(len(unique_vals)-1):
            val = unique_vals[j+1]
            # If the mask is of a different honeycomb then the currently segmented
            if val != i+1:
                # Extract the mask and do closing 
                valMask = np.zeros(unfolded_helperStack.shape)
                valMask[unfolded_helperStack==val]=1
                valMask = closing(valMask,selem=np.ones((1,7,7)))
                # Find indices of the mask and add them to the new stack
                valPos = np.array(np.where(valMask==1))
                unfolded_helperStackFixed[valPos[0,:],valPos[1,:],valPos[2,:]] = 1
                # Limit the indices to have 1 value in each column
                _, uniqueColIdx = np.unique(valPos[[0,2],:],return_index=True,axis=1)
                valPos = valPos[:,uniqueColIdx]
                # Loop through columns and fill the new stack up or down depending on positions
                meanCurrHeight = np.mean(valPos[1,:])
                for k in range(valPos.shape[1]):
                    # if unfolded_helperStackFixed[valPos[0,k],valPos[1,k],valPos[2,k]] == 0:
                    if meanCurrHeight > meanLayerHeight:
                        idxLen = unfolded_helperStack.shape[1] - valPos[1,k] + 3
                        idxList = (np.repeat(valPos[0,k],idxLen),
                                            np.arange(valPos[1,k]-3,unfolded_helperStack.shape[1],1),
                                            np.repeat(valPos[2,k],idxLen))
                    else:
                        idxLen = valPos[1,k] + 1 + 3
                        idxList = (np.repeat(valPos[0,k],idxLen),
                                            np.arange(0,valPos[1,k] + 1 + 3,1),
                                            np.repeat(valPos[2,k],idxLen))
                    unfolded_helperStackFixed[idxList] = 1

        # Calculate cost
        honeycombCost_stack = (1 - helpers.sigmoidProbFunction(unfolded_stack,means,variances, weight=0.5, visualize=visualizeUnfolding))*255
        backgroundCost_stack = helpers.sigmoidProbFunction(unfolded_stack,means,variances, weight=0.001, visualize=visualizeUnfolding)*255
        
        zstackList = []
        for j in range(unfolded_stack.shape[0]):        
            unfolded_img = unfolded_stack[j,:,:]
            unfolded_helper = unfolded_helperStackFixed[j,:,:]

            honeycombCost = honeycombCost_stack[j,:,:]
            backgroundCost = backgroundCost_stack[j,:,:]

            backgroundCost[unfolded_helper>0] = 300

            if visualizeUnfolding == True and j == 0:
                plt.figure()
                plt.imshow(backgroundCost)
                plt.show()

            unfolded_img = helpers.rescaleImage(unfolded_img, 1, 255)

            ### Layered surfaces detection
            segmentation_surfaces = unfoldedHoneycombSurfDetect(unfolded_img, honeycombCost, backgroundCost,
                visualize=(visualizeUnfolding), return_helper_surfaces=returnHelperSurfaces, darkSurfWeight=100)

            ## Fold detected lines back to original image shape
            folded_surfaces = hc.fold_surfaces_back(segmentation_surfaces, zIdx=j)
            zstackList.append(folded_surfaces)
            print(f"Layer {i+1}, Zstack {j+1}")
        layersList.append(zstackList)
        print(f"Finished Layer {i+1} for the final detection")
    return layersList


if __name__ == "__main__":

    # I = skimage.io.imread('data/29-2016_29-2016-60kV-resized_z200.tif')
    # I = skimage.io.imread('data/29-2016_29-2016-60kV-resized.tif')
    # I = skimage.io.imread('data/29-2016_29-2016-60kV-zoom-center_recon.tif')
    # I = skimage.io.imread('data/29-2016_29-2016-60kV-LFoV-center-+-1-ring_recon.tif')
    I = skimage.io.imread('data/NL07C_NL07C-60kV-LFoV-center-+-1-ring_recon.tif')

    I = I[390:640,:,:]
    # I = I[450:470,:,:]
    # I = I[390:410,:,:]


    # I_2d = I[200,:,:]

    # Rotation (if needed)
    Inew = []
    for i in range(I.shape[0]):
        Inew.append(scp.rotate(I[i,:,:],-15))
    I = np.array(Inew)

    visualizeUnfolding = False # if True - visualizes the unfolding process steps
    interpStep = 1 # Distance between the interpolated points
    normalLinesRange = 15 # Range (half of the length) of lines normal to interpolation points
    normalLinesNumPoints = 60 # Number of interpolation points along a normal line
    returnHelperSurfaces = False # if True, returns also dark helper surfaces from surface detection process
    params = [visualizeUnfolding, interpStep, normalLinesRange, normalLinesNumPoints,
         returnHelperSurfaces]
    # savePointsPath="data/cornerPoints/NLbig_z390-640_rot_-15.txt"
    # savePointsPath="data/cornerPoints/test.txt"
    savePointsPath=""
    # loadPointsPath=""
    # loadPointsPath="data/cornerPoints/NLbig_z390-640_rot_-15.txt"
    loadPointsPath="data/cornerPoints/test.txt"

    layersList = detect3dSlicewiseLayers(I, 2, params, savePointsPath=savePointsPath, loadPointsPath=loadPointsPath)
    
    # plt.figure()
    # plt.imshow(I[1,:,:], cmap='gray')
    # for i in range(len(layersList)):
    #     folded_surfaces = layersList[i][1]
    #     for j in range(len(folded_surfaces)):
    #         folded_surface = folded_surfaces[j]
    #         if j < 2:
    #             plt.plot(folded_surface[0,:],folded_surface[1,:], 'r')
    #         else:
    #             plt.plot(folded_surface[0,:],folded_surface[1,:], 'b')

    # plt.show()

    # segmImg = helpers.layersToMatrix(layersList, I.shape)

    # plt.imshow(segmImg[15,:,:].transpose())
    # plt.show()

    # surf = helpers.layersToSurface(layersList)
    # surf_array = np.array(surf)
    # np.save("data/NLbig_slicewise_z390-640_allSurf_raw_2.npy", surf_array)

    # # vwl.save_multSurf2vtk('data/surfaces/slicewise_z200New_allSurf.vtk', surf)
