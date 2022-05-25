# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 17:10:52 2022

@author: Pawel Pieta s202606@student.dtu.dk
"""

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage.io 
import scipy.ndimage as scp
import copy
from skimage.morphology import closing
import os

from honeycomb.unfolding.unfold2d import Unfold2d
from honeycomb.surface_detection import surfaceDetector2d
from honeycomb.helpers import misc
from honeycomb import cornerPointsAgent

import time


class SegmentationPipeline:
    """Performs layered surfaces honeycomb wall edges detection on a single slice."""

    def __init__(self, imgSlice, wallDetector=None, helperDetector=None, interpStep=1, normalLinesRange=20, normalLinesNumPoints=40, returnHelperSurfaces=False, a_parabola=0.05, wallCostWeight=0.5, helperCostWeight=0.001):
        """Class initialization.\n
        Params:\n
        imgSlice - 2D honecomb image slice\n
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
        self.imgSlice = imgSlice
        self.wallDetector = wallDetector
        self.helperDetector = helperDetector
        self.interpStep = interpStep
        self.normalLinesRange = normalLinesRange
        self.normalLinesNumPoints = normalLinesNumPoints
        self.returnHelperSurfaces = returnHelperSurfaces

        self.wallCostWeight = wallCostWeight
        self.helperCostWeight = helperCostWeight

        self.vis_img = np.copy(imgSlice)
        self.vis_img = misc.rescaleImage(self.vis_img, 1, 255)

        ## Calculate image gaussian model 
        self.means, self.variances = misc.imgGaussianModel(imgSlice)

        # Prepare parabola vector
        self.parVec = np.arange(-normalLinesNumPoints/2,normalLinesNumPoints/2,1)
        self.parVec = a_parabola*self.parVec**2+1

        self.hcList = []
        self.hcHelpList = []
        self.layersList = []

    def unfoldSlice(self):
        """Loops through the honeycomb wall objects and performs full unfolding process.
        """

        for i in range(len(self.hcList)):
            hc = self.hcList[i]
            misc.print_statusline(f"Unfolding of wall {i+1} started")
            hc.interpolate_points(step=self.interpStep)
            misc.print_statusline(f"Wall {i+1}, points interpolated")
            hc.smooth_interp_corners()
            misc.print_statusline(f"Wall {i+1}, points smoothed")
            t0 = time.time()
            hc.calculate_normals(normals_range=self.normalLinesRange)
            t1 = time.time()
            misc.print_statusline(f"Wall {i+1}, normals calculated")
            t0 = time.time()
            hc.get_unfold_points_from_normals(interp_points=self.normalLinesNumPoints)
            t1 = time.time()
            misc.print_statusline(f"Wall {i+1}, normals interpolated")
            hc.unfold_image()
            misc.print_statusline(f"Wall {i+1}, image unfolded")
            self.hcList[i] = hc

    def detectHelperWallCenter(self, visualize=True):
        """Detects the approximation of the center of the honeycomb wall. 
        Used for modification of the cost fuction at the final segmenatation step\n
        Params:\n
        visualize - If True - shows the results of detection
        """

        helperSlice = np.zeros(self.imgSlice.shape)

        for i in range(len(self.hcList)):
            #Get unfolded slice
            hc = self.hcList[i]
            unfolded_img = hc.unfolded_img

            # Calculate cost
            helperCost = (1 - misc.sigmoidProbFunction(unfolded_img,self.means,self.variances, weight=self.helperCostWeight, visualize=visualize))*255
            helperCost = scp.uniform_filter(helperCost,size=3)
            # Add the parabola
            helperCost = np.moveaxis((np.moveaxis(helperCost,0,-1)+self.parVec),-1,0)
            
            misc.print_statusline(f"Helper surface, wall {i+1}")

            #calculate helper line
            helperSurf = self.helperDetector.detect(helperCost, visualize=visualize)

            ## Fold detected lines back to original image shape
            folded_helper = np.round(hc.fold_surfaces_back(helperSurf)[0]).astype('int')
            #Apply detection to a helper stack
            helperSlice[folded_helper[1,:],folded_helper[0,:]]=i+1

            misc.print_statusline(f"Finished wall {i+1} for the helper detection")

        # Make a copy of the hc objects with different image
        self.hcHelpList = copy.deepcopy(self.hcList)
        for i in range(len(self.hcList)):
            hcHelp = self.hcHelpList[i]
            hcHelp.img = helperSlice
            hcHelp.unfold_image(method='nearest')
            self.hcHelpList[i] = hcHelp
            misc.print_statusline(f"Helper image {i} unfolded")

    def detectWallEdges(self, visualize=True):
        """Performs the final detection of the honeycomb wall edges for all marked walls.
        Params:\n
        visualize - If True - shows the results of detection
        """

        if self.wallDetector is None:
            misc.print_statusline("Wall detector was not defined, segmentation can't be performed.")
            return

        for i in range(len(self.hcList)):
            misc.print_statusline(f"Final detection, wall {i+1} started")
            hc = self.hcList[i]
            unfolded_img = hc.unfolded_img

            if self.helperDetector is not None:
                hcHelp = self.hcHelpList[i]
                unfolded_helper = hcHelp.unfolded_img
                # Fix the helper to be consistent
                meanLayerHeight = np.mean(np.where(unfolded_helper==i+1)[0])
                unfolded_helperFixed = np.zeros(unfolded_helper.shape)
                unique_vals = np.unique(unfolded_helper)
                # Loop through masks of the honeycomb layers
                for j in range(len(unique_vals)-1):
                    val = unique_vals[j+1]
                    # If the mask is of a different honeycomb then the currently segmented
                    if val != i+1:
                        # Extract the mask and do closing 
                        valMask = np.zeros(unfolded_helper.shape)
                        valMask[unfolded_helper==val]=1
                        valMask = closing(valMask,selem=np.ones((7,7)))
                        # Find indices of the mask and add them to the new stack
                        valPos = np.array(np.where(valMask==1))
                        unfolded_helperFixed[valPos[0,:],valPos[1,:]] = 1
                        # Limit the indices to have 1 value in each column
                        _, uniqueColIdx = np.unique(valPos[1,:],return_index=True)
                        valPos = valPos[:,uniqueColIdx]
                        # Loop through columns and fill the new stack up or down depending on positions
                        meanCurrHeight = np.mean(valPos[0,:])
                        for k in range(valPos.shape[1]):
                            # if unfolded_helperStackFixed[valPos[0,k],valPos[1,k],valPos[2,k]] == 0:
                            if meanCurrHeight > meanLayerHeight:
                                idxLen = unfolded_helper.shape[0] - valPos[0,k] + 2
                                idxList = (np.arange(valPos[0,k]-2,unfolded_helper.shape[0],1),np.repeat(valPos[1,k],idxLen))
                            else:
                                idxLen = valPos[0,k] + 1 + 2
                                idxList = (np.arange(0,valPos[0,k] + 1 + 2,1),np.repeat(valPos[1,k],idxLen))
                            unfolded_helperFixed[idxList] = 1

            # Calculate cost
            honeycombCost = (1 - misc.sigmoidProbFunction(unfolded_img,self.means,self.variances, weight=self.wallCostWeight, visualize=visualize))*255
            backgroundCost = misc.sigmoidProbFunction(unfolded_img,self.means,self.variances, weight=self.helperCostWeight, visualize=visualize)*255
            
            backgroundCost = np.moveaxis((np.moveaxis(backgroundCost,0,-1)+self.parVec),-1,0)

            if self.helperDetector is not None:
                unfolded_helper = unfolded_helperFixed[:,:]
                backgroundCost[unfolded_helper>0] = 300

            if visualize == True:
                plt.figure()
                ax = plt.gca()
                im = plt.imshow(backgroundCost)
                plt.axis('off')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                
                plt.colorbar(im, cax=cax)
                plt.show()

            unfolded_img = misc.rescaleImage(unfolded_img, 1, 255)

            ### Layered surfaces detection
            segmentation_surfaces = self.wallDetector.detect(unfolded_img, honeycombCost, backgroundCost,
                visualize=visualize, return_helper_surfaces=self.returnHelperSurfaces)
            ## Fold detected lines back to original image shape
            folded_surfaces = hc.fold_surfaces_back(segmentation_surfaces)
            self.layersList.append(folded_surfaces)
            misc.print_statusline(f"Finished wall {i+1} for the final detection")

    def segmentVolume(self, wallNum, savePointsPath='', loadPointsPath='', visualize=True):
        """ Performs all the steps needed for detection of the wall edges in a slice of the honeycomb scan.\n
        Parameters:\n
        wallNum - number of honeycomb walls to detect\n
        savePointsPath - if not '', saves manually marked points as a txt file
        loadPointsPath - if not '', loads the points from given path instead of initializing manual marking.
        """
        
        # Preparing separate unfolding objects for each wall
        self.hcList = [Unfold2d(self.imgSlice, self.vis_img, visualize=visualize) for i in range(wallNum)]

        #Defining the corner points by loading or manual pointing
        if loadPointsPath == '':
            for hc in self.hcList:
                self.vis_img = hc.draw_corners()
        else:
            self.hcList = cornerPointsAgent.load2dHcPoints(loadPointsPath, self.hcList)

        # Saving the points
        if savePointsPath != '':
            cornerPointsAgent.save2dHcPoints(savePointsPath, self.hcList)

        # Unfolding the wall images
        self.unfoldSlice()

        if self.helperDetector is not None:
            self.detectHelperWallCenter(visualize=visualize)

        # Final detection
        self.detectWallEdges(visualize=visualize)

        misc.print_statusline(f"2D honeycomb wall segmentation finished.")
       
        return self.layersList


if __name__ == "__main__":

    # I = skimage.io.imread('data/29-2016_29-2016-60kV-zoom-center_recon.tif')
    I = skimage.io.imread('data/NL07C_NL07C-60kV-LFoV-center-+-1-ring_recon.tif')

    # I_2d = I[200,:,:]
    I_2d = I[390,:,:]
    # Rotate if needed
    I_2d = scp.rotate(I_2d,-15)

    ####### Params #######
    visualize = False # if True - visualizes the unfolding process steps
    #### Segmentation params
    wallNum = 2
    # savePointsPath = "data/cornerPoints/H29slicewise_z200.txt"
    savePointsPath = ""
    # loadPointsPath= ""
    # loadPointsPath= "data/cornerPoints/H29oneSlice_thesisViz_z200.txt"
    loadPointsPath= "data/cornerPoints/NLbigOneSlice_2wallSegments_z390.txt"
    #### Unfolding params
    interpStep = 1 # Distance between the interpolated points
    normalLinesRange = 15 # Range (half of the length) of lines normal to interpolation points
    normalLinesNumPoints = 60 # Number of interpolation points along a normal line
    #### Detection params
    # In segmentation
    returnHelperSurfaces = False # if True, returns also dark helper surfaces from surface detection process
    a_parabola = 0.05 # a parameter of y=ax^2+b equation, used to modify the helper detection cost function
    # Helper
    helperDetectionSmoothness = 1 # how much in y direction the line can move with each step in the x direction (only int values)
    # Main detection
    edgeSmoothness=1 
    helperSmoothness=1 
    helperWeight=100 # multiplier of how much more "important" the helper line detetion is
    wallThickness=[4,16] # min-max value of the distance between teh edges of the wall
    darkHelperDist=[6, 20] # min-max distance between the "dark" helper lines following the background path
    darkWhiteHelperDist=[2,10] # min-max distance between a "dark" helper line and the wall central helper line
    # Cost function
    wallCostWeight = 0.5 # defines the position of 0.5 probability value in the wall cost function, if set to 0.5 it is in the middle between the means of the distributions
    helperCostWeight = 0.001 # same as above, applies both to helper detection and to helper surfaces in the main detection

    # Main wall detector instance
    wallDetector = surfaceDetector2d.WallEdgeDetector(edgeSmoothness=edgeSmoothness, 
                                                    helperSmoothness=helperSmoothness, 
                                                    helperWeight=helperWeight, 
                                                    wallThickness=wallThickness,
                                                    darkHelperDist=darkHelperDist, 
                                                    darkWhiteHelperDist=darkWhiteHelperDist)
    # Helper wall center detector instance
    helperDetector = surfaceDetector2d.WallCenterDetector(smoothness=helperDetectionSmoothness)
    # 2D segmentation instance
    pipeline = SegmentationPipeline(imgSlice=I_2d, 
                                    wallDetector=wallDetector, 
                                    helperDetector=helperDetector, 
                                    interpStep=interpStep, 
                                    normalLinesRange=normalLinesRange, 
                                    normalLinesNumPoints=normalLinesNumPoints, 
                                    returnHelperSurfaces=returnHelperSurfaces, 
                                    a_parabola=a_parabola,
                                    wallCostWeight=wallCostWeight,
                                    helperCostWeight=helperCostWeight)
# Run the segmentation
    layersList = pipeline.segmentVolume(wallNum=wallNum, 
                                        savePointsPath=savePointsPath, 
                                        loadPointsPath=loadPointsPath, 
                                        visualize=visualize)

    plt.figure()
    plt.imshow(I_2d, cmap='gray')
    for i in range(len(layersList)):
        folded_surfaces = layersList[i]
        for j in range(len(folded_surfaces)):
            folded_surface = folded_surfaces[j]
            if j < 2:
                plt.plot(folded_surface[0,:],folded_surface[1,:], 'r')
            else:
                plt.plot(folded_surface[0,:],folded_surface[1,:], 'b')

    plt.show()