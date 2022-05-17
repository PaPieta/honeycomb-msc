# -*- coding: utf-8 -*-
"""
Created on Fri May 06 17:50:16 2022

@author: Pawel Pieta s202606@student.dtu.dk
"""

import numpy as np 
import matplotlib.pyplot as plt
import skimage.io 
import scipy.ndimage as scp
import copy
from skimage.morphology import closing
import os

from honeycomb.unfolding.unfold3d import Unfold3d
from honeycomb.surface_detection import surfaceDetector3d
from honeycomb.helpers import misc
from honeycomb.helpers import vtk_write_lite as vwl # for saving vtk file

import time


class SegmentationPipeline:
    """Performs layered surfaces honeycomb wall edges detection on a whole stack of images using a slicewise approach."""

    def __init__(self, imgStack, wallDetector, helperDetector=None, interpStep=1, normalLinesRange=20, normalLinesNumPoints=40, returnHelperSurfaces=False, a_parabola=0.05, wallCostWeight=0.5, helperCostWeight=0.001):
        """Class initialization.\n
        Params:\n
        imgStack - 3D honecomb image stack\n
        wallDetector - WallEdgeDetector class object with correctly defined parameters\n
        helperDetector - WallCenterDetector class if helper detection is to be performed. Else, set to None\n
        Unfolding params:\n
        interpStep - distance between the interpolated points\n
        normalLinesRange - Range (half of the length) of lines normal to interpolation points\n
        normalLinesNumPoints - Number of interpolation points along a normal line\n
        Detection parameters:\n
        returnHelperSurfaces - if True, the method will also return the helper surfaces detected during segmentation\n
        a_parabola - a parameter of y=ax^2+b equation, used to modify the helper detection cost function (to encoruage the line to stay in the center)\n
        wallCostWeight - defines the position of 0.5 probability value in the wall cost function, if set to 0.5 it is in the middle between the means of the distributions\n
        helperCostWeight - same as above, applies both to helper detection and to helper surfaces in the main detection
        """
        self.imgStack = imgStack
        self.wallDetector = wallDetector
        self.helperDetector = helperDetector
        self.interpStep = interpStep
        self.normalLinesRange = normalLinesRange
        self.normalLinesNumPoints = normalLinesNumPoints
        self.returnHelperSurfaces = returnHelperSurfaces

        self.wallCostWeight = wallCostWeight
        self.helperCostWeight = helperCostWeight

        self.vis_img = np.copy(imgStack)
        self.vis_img = misc.rescaleImage(self.vis_img, 1, 255)

        ## Calculate image gaussian model 
        self.means, self.variances = misc.imgGaussianModel(imgStack)

        # Prepare parabola vector
        self.parVec = np.arange(-normalLinesNumPoints/2,normalLinesNumPoints/2,1)
        self.parVec = a_parabola*self.parVec**2+1

        self.hcList = []
        self.hcHelpList = []
        self.layersList = []
        
    def saveHcPoints(self, savePath):
        """Private function. Used for saving the points marked by the user into a txt file\n
        Params:\n
        savePath - Full file save path
        """

        saveDir = os.path.dirname(savePath)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        with open(savePath, 'w') as f:
            for i in range(len(self.hcList)):
                hc = self.hcList[i]
                for j in range(len(hc.lines)):
                    line = hc.lines[j]
                    f.write(f"HC {i} {j} {line.shape[1]}\n")
                    for k in range(line.shape[1]):
                        f.write(f"{line[0,k]} {line[1,k]} {line[2,k]}\n")

    def loadHcPoints(self, loadPath):
        """Private function. Used for loading from txt file ponts previously marked by the user.\n
        Params:\n
        savePath - Full file load path
        """

        with open(loadPath, "r") as f:
            contents = f.readlines()

        hcIdx = -1
        for i in range(len(contents)):
            currString = contents[i]
            elemList = currString.split()
            if "HC" in currString:
                prevHcIdx = hcIdx
                hcIdx =  int(elemList[1])
                if prevHcIdx != hcIdx:
                    if prevHcIdx != -1:
                        lineList.append(line)
                        self.hcList[prevHcIdx].lines = lineList
                    lineList = []
                else:
                    lineList.append(line)
                lineIdx =  int(elemList[2])
                numPoints =  int(elemList[3])
                pointsCounter = 0
                line = np.zeros((3,numPoints))
            elif currString != '\n':
                line[:,pointsCounter] = np.array([float(elemList[0]), float(elemList[1]), float(elemList[2])])
                pointsCounter += 1
        
        lineList.append(line)
        self.hcList[hcIdx].lines = lineList

    def unfoldStack(self):
        """Private function. Loops through the honeycomb wall objects and performs full unfolding process.
        """

        for i in range(len(self.hcList)):
            hc = self.hcList[i]
            print(f"Unfolding of layer {i+1} started")
            hc.interpolate_points(step=self.interpStep)
            print("Points interpolated")
            hc.smooth_interp_corners()
            print("Points smoothed")
            t0 = time.time()
            hc.calculate_normals(normals_range=self.normalLinesRange)
            t1 = time.time()
            print(f"Normals calculated, time: {t1-t0}")
            t0 = time.time()
            hc.get_unfold_points_from_normals3(interp_points=self.normalLinesNumPoints)
            t1 = time.time()
            print(f"Normals interpolated, time: {t1-t0}")
            hc.unfold_image()
            print("Image unfolded")
            self.hcList[i] = hc

    def detectHelperWallCenter(self, visualize=True):
        """Private function. Detects the approximation of the center of the honeycomb wall. 
        Used for modification of the cost fuction at the final segmenatation step\n
        Params:\n
        visualize - If True - shows the results of detection
        """

        helperStack = np.zeros(self.imgStack.shape)

        for i in range(len(self.hcList)):
            #Get unfolded stack
            hc = self.hcList[i]
            unfolded_stack = hc.unfolded_img

            # Calculate cost
            helperCost_stack = (1 - misc.sigmoidProbFunction(unfolded_stack,self.means,self.variances, weight=self.helperCostWeight, visualize=visualize))*255
            helperCost_stack = scp.uniform_filter(helperCost_stack,size=3)
            # Add the parabola
            helperCost_stack = np.moveaxis((np.moveaxis(helperCost_stack,1,-1)+self.parVec),-1,1)

            helperSurf = self.helperDetector.detect(helperCost_stack, visualize=visualize, builderType='default')
            
            ## Fold detected lines back to original image shape
            folded_helper = np.round(hc.fold_3d_surfaces_back(helperSurf)[0]).astype('int')
            helperStack[folded_helper[2,:],folded_helper[1,:],folded_helper[0,:]]=i+1
            print(f"Finished layer {i+1} for the helper detection")

        # Make a copy of the hc objects with different image
        self.hcHelpList = copy.deepcopy(self.hcList)
        for i in range(len(self.hcList)):
            hcHelp = self.hcHelpList[i]
            hcHelp.img = helperStack
            hcHelp.unfold_image(method='nearest')
            self.hcHelpList[i] = hcHelp
            print(f"Helper image {i} unfolded")

    def detectWallEdges(self, visualize=True):
        """Private function. Performs the final detection of the honeycomb wall edges for all marked walls.
        Params:\n
        visualize - If True - shows the results of detection
        """

        for i in range(len(self.hcList)):
            print(f"Final detection, layer {i+1} started")
            hc = self.hcList[i]
            unfolded_stack = hc.unfolded_img

            if self.helperDetector is not None:
                hcHelp = self.hcHelpList[i]
                unfolded_helperStack = hcHelp.unfolded_img
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
                                idxLen = unfolded_helperStack.shape[1] - valPos[1,k] + 2
                                idxList = (np.repeat(valPos[0,k],idxLen),
                                                    np.arange(valPos[1,k]-2,unfolded_helperStack.shape[1],1),
                                                    np.repeat(valPos[2,k],idxLen))
                            else:
                                idxLen = valPos[1,k] + 1 + 2
                                idxList = (np.repeat(valPos[0,k],idxLen),
                                                    np.arange(0,valPos[1,k] + 1 + 2,1),
                                                    np.repeat(valPos[2,k],idxLen))
                            unfolded_helperStackFixed[idxList] = 1

            # Calculate cost
            honeycombCost_stack = (1 - misc.sigmoidProbFunction(unfolded_stack,self.means,self.variances, weight=self.wallCostWeight, visualize=visualize))*255
            backgroundCost_stack = misc.sigmoidProbFunction(unfolded_stack,self.means,self.variances, weight=self.helperCostWeight, visualize=visualize)*255
            
            backgroundCost_stack = np.moveaxis((np.moveaxis(backgroundCost_stack,1,-1)+self.parVec),-1,1)

            if self.helperDetector is not None:
                # unfolded_helper = unfolded_helperStackFixed[j,:,:]
                backgroundCost_stack[unfolded_helperStackFixed>0] = 300

            if visualize == True and j == 0:
                plt.figure()
                plt.imshow(backgroundCost_stack[0,:,:])
                plt.show()

            unfolded_stack = misc.rescaleImage(unfolded_stack, 1, 255)

            ### Layered surfaces detection
            t0 = time.time()
            segmentation_surfaces = self.wallDetector.detect(unfolded_stack, honeycombCost_stack, backgroundCost_stack,
                visualize=visualize, return_helper_surfaces=self.returnHelperSurfaces, builderType='parallel_6')
            t1 = time.time()
            ## Fold detected lines back to original image shape
            folded_surfaces = hc.fold_3d_surfaces_back(segmentation_surfaces, representation='matrix')
            self.layersList.append(folded_surfaces)
            
            print(f"Finished Layer {i+1} for the final detection, segmentation time: {t1-t0}")


    def segmentVolume(self, layerNum, savePointsPath='', loadPointsPath='', visualize=True):
        """ Performs all the steps needed for detection of the wall edges in the honeycomb scan.\n
        Parameters:\n
        layerNum - number of layers to detect\n
        savePointsPath - if not '', saves manually marked points as a txt file
        loadPointsPath - if not '', loads the points from given path instead of initializing manual marking.
        """
        
        # Preparing separate unfolding objects for each wall
        self.hcList = [Unfold3d(self.imgStack, self.vis_img, visualize=visualize) for i in range(layerNum)]

        #Defining the corner points by loading or manual pointing
        if loadPointsPath == '':
            for hc in self.hcList:
                self.vis_img = hc.draw_corners()
        else:
            self.loadHcPoints(loadPointsPath)

        # Saving the points
        if savePointsPath != '':
            self.saveHcPoints(savePointsPath)

        # Unfolding the wall images
        self.unfoldStack()

        # # Optional save of manually marked mesh
        # manualSurf = []
        # for hc in self.hcList:
        #     manualSurf.append(np.moveaxis(hc.lines_interp,0,-1))
        # np.save("data/H29big_slicewise_z380-620_allSurf_manual.npy", np.array(manualSurf, dtype=object))

        if self.helperDetector is not None:
            self.detectHelperWallCenter(visualize=visualize)

        # Final detection
        self.detectWallEdges(visualize=visualize)
       
        return self.layersList


if __name__ == "__main__":
    # TODO: Add cost function weight as the parameter, try decreasing the number of points along a normal line
    I = skimage.io.imread('data/29-2016_29-2016-60kV-zoom-center_recon.tif')
    # I = skimage.io.imread('data/29-2016_29-2016-60kV-LFoV-center-+-1-ring_recon.tif')
    # I = skimage.io.imread('data/NL07C_NL07C-60kV-LFoV-center-+-1-ring_recon.tif')
    # I = skimage.io.imread('data/NL07C_NL07C-60kV-zoom-center_recon.tif')
    # I = skimage.io.imread('data/PD27A-60kV-zoom-center_recon.tif')
    # I = skimage.io.imread('data/PB27A-60kV-LFoV-center-+-1-ring_recon.txm.tif')

    I = I[200:250]
    # I = I[:100,:,:]

    # Rotation (if needed)
    # Inew = []
    # for i in range(I.shape[0]):
    #     Inew.append(scp.rotate(I[i,:,:],-15))
    # I = np.array(Inew)

    ####### Params #######
    visualizeUnfolding = False # if True - visualizes the unfolding process steps
    #### Segmentation params
    layerNum = 1
    # savePointsPath="data/cornerPoints/NLbig_z390-640_rot_-15.txt"
    # savePointsPath="data/cornerPoints/H29big_slicewise_z380-620.txt"
    # loadPointsPath="data/cornerPoints/NLbig_z390-640_rot_-15.txt"
    # loadPointsPath="data/cornerPoints/H29big_slicewise_z380-620.txt"
    # savePointsPath = "data/cornerPoints/PD_z0-950_rot_-15.txt"
    # savePointsPath = "data/cornerPoints/PBbig_z250-780_rot_-15.txt"
    savePointsPath = ""
    loadPointsPath = "data/cornerPoints/H29_z200-250_1surf.txt"
    #### Unfolding params
    interpStep = 10/4 # Distance between the interpolated points
    normalLinesRange = 30 # Range (half of the length) of lines normal to interpolation points
    normalLinesNumPoints = 60 # Number of interpolation points along a normal line
    #### Detection params
    # In segmentation
    returnHelperSurfaces = False # if True, returns also dark helper surfaces from surface detection process
    a_parabola = 0.05 # a parameter of y=ax^2+b equation, used to modify the helper detection cost function
    # Helper
    helperDetectionSmoothness = 1 # how much in y direction the line can move with each step in the x direction (only int values)
    # Main detection
    edgeSmoothness=2 
    helperSmoothness=1 
    helperWeight=180 # multiplier of how much more "important" the helper line detetion is
    wallThickness=[6,25] # min-max value of the distance between teh edges of the wall
    darkHelperDist=[12, 35] # min-max distance between the "dark" helper lines following the background path
    darkWhiteHelperDist=[1,30] # min-max distance between a "dark" helper line and the wall central helper line
    # Cost function
    wallCostWeight = 0.5 # defines the position of 0.5 probability value in the wall cost function, if set to 0.5 it is in the middle between the means of the distributions
    helperCostWeight = 0.001 # same as above, applies both to helper detection and to helper surfaces in the main detection

    # Main wall detector instance
    wallDetector = surfaceDetector3d.WallEdgeDetector(edgeSmoothness=edgeSmoothness, 
                                                    helperSmoothness=helperSmoothness, 
                                                    helperWeight=helperWeight, 
                                                    wallThickness=wallThickness,
                                                    darkHelperDist=darkHelperDist, 
                                                    darkWhiteHelperDist=darkWhiteHelperDist)
    # Helper wall center detector instance
    helperDetector = surfaceDetector3d.WallCenterDetector(smoothness=helperDetectionSmoothness)
    # Slicewise segmentation instance
    pipeline = SegmentationPipeline(imgStack=I, 
                                    wallDetector=wallDetector, 
                                    helperDetector=None, 
                                    interpStep=interpStep, 
                                    normalLinesRange=normalLinesRange, 
                                    normalLinesNumPoints=normalLinesNumPoints, 
                                    returnHelperSurfaces=returnHelperSurfaces, 
                                    a_parabola=a_parabola,
                                    wallCostWeight=wallCostWeight,
                                    helperCostWeight=helperCostWeight)
    # Run the segmentation
    layersList = pipeline.segmentVolume(layerNum=layerNum, 
                                        savePointsPath=savePointsPath, 
                                        loadPointsPath=loadPointsPath, 
                                        visualize=visualizeUnfolding)
    
    # Plot the results
    plt.figure()
    plt.imshow(I[1,:,:], cmap='gray')
    for i in range(len(layersList)):
        folded_surfaces = layersList[i]
        for j in range(len(folded_surfaces)):
            folded_surface = folded_surfaces[j][:2,1,:]
            if j < 2:
                plt.plot(folded_surface[0,:],folded_surface[1,:], 'r')
            else:
                plt.plot(folded_surface[0,:],folded_surface[1,:], 'b')

    plt.show()

    # segmImg = helpers.layersToMatrix(layersList, I.shape)

    # plt.imshow(segmImg[15,:,:].transpose())
    # # plt.show()

    # surf = helpers.layersToSurface(layersList)
    # surf_array = np.array(surf)
    # np.save("data/H29_slicewise_z200-780_allSurf_raw.npy", surf_array)

    # vwl.save_multSurf2vtk('data/surfaces/slicewise_z200New_allSurf.vtk', surf)
