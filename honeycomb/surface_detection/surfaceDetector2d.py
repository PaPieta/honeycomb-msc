# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 17:21:39 2022

@author: Pawel Pieta s202606@student.dtu.dk
"""

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import slgbuilder


class WallCenterDetector:
    """Simple detector for finding a  rough approximation of a central line/surface in a honeycomb wall. Used for modifying the cost function of the helper lines before the main detection."""

    def __init__(self, smoothness=1):
        """Class initialization.\n
        Params:\n
        smoothness - how much in y direction the line can move with each step in the x direction (only int values)
        """
        self.smoothness = smoothness

    def detect(self, honeycombCost, visualize=True, builderType = 'default'):
        """Performs central surface detection on the provided cost image
        Params:\n
        honeycombCost - 2D cost function image for the honeycomb-targeting surfaces.\n
        visualize - if True, will visualize the result of detection\n
        builderType - default or parallel_{2,4,6}, number represents the number of used cores.
        """

        layers = [slgbuilder.GraphObject(honeycombCost)]
        if builderType == 'default':
            helper = slgbuilder.MaxflowBuilder(capacity_type=np.float64)
        elif builderType == 'parallel_2':
            helper = slgbuilder.PBKBuilder(capacity_type=np.float64, num_threads=2, expected_blocks=1)
        elif builderType == 'parallel_4':
            helper = slgbuilder.PBKBuilder(capacity_type=np.float64, num_threads=4, expected_blocks=1)
        elif builderType == 'parallel_6':
            helper = slgbuilder.PBKBuilder(capacity_type=np.float64, num_threads=6, expected_blocks=1)
        else:
            raise AttributeError("Wrong builder type, has to be either 'default' or 'parallel_{2,4,6}'")
        helper.add_objects(layers)
        
        helper.add_layered_boundary_cost() 

        helper.add_layered_smoothness(layers,delta=self.smoothness, wrap=True)

        ## Cut
        helper.solve()
        segmentations = [helper.get_labels(l).astype(np.int32) for l in layers]
        segmentation_lines = [s.shape[0] - np.argmax(s[::-1,:], axis=0) - 1 for s in segmentations]
        #Correct for 0.5 pixel shift
        segmentation_lines = [x+0.5 for x in segmentation_lines]

        ## Unfolded image visualization
        if visualize == True:
            plt.figure()
            ax = plt.gca()
            im = plt.imshow(honeycombCost)
            plt.axis('off')
            for i in range(len(segmentation_lines)):
                    plt.plot(segmentation_lines[i], 'r')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()

        return segmentation_lines


class WallEdgeDetector:
    """Detector for finding surfaces describing the edges of the honeycomb wall."""

    def __init__(self, edgeSmoothness=1, helperSmoothness=1, helperWeight=100, wallThickness=[4,16], darkHelperDist=[6, 20], darkWhiteHelperDist=[1,10]):
        """Class initialization.\n
        Params:\n
        edgeSmoothness - smoothness of the wall surfaces: how much in y direction the line can move with each step in the x direction (only int values)\n
        helperSmoothness - smoothness of the helper surfaces\n
        helperWeight - multiplier of how much more "important" the helper line detetion is, can limit overlapping where the neighbouring walls touch\n
        wallThickness - min-max value of the distance between teh edges of the wall\n
        darkHelperDist - min-max distance between the "dark" helper lines following the background path\n
        darkWhiteHelperDist - min-max distance between a "dark" helper line and the wall central helper line\n
        visualize - if True, will visualize the result of detection
        """
        self.edgeSmoothness = edgeSmoothness
        self.helperSmoothness = helperSmoothness
        self.helperWeight = helperWeight
        self.wallThickness = wallThickness
        self.darkHelperDist = darkHelperDist
        self.darkWhiteHelperDist = darkWhiteHelperDist

    def detect(self, unfolded_img, honeycombCost, backgroundCost, visualize=False, return_helper_surfaces=False, builderType = 'default'):
        """ Performs detection of the honeycomb wall edges.\n
        Params:\n
        unfolded_img - clean unfolded image
        honeycombCost - cost function image for the honeycomb-targeting surfaces \n
        backgroundCost - cost function image for the background-targeting surfaces \n
        visualize - if True, plots the resulting surfaces on an unfolded image \n
        return_helper_surfaces - if True, returns also dark helper surfaces.\n
        builderType - default or parallel_{2,4,6}, number represents the number of used cores.
        """

        darkToHoneycombMinMargin = 1 # min distance between the helper surface and honeycomb edge

        layers = [slgbuilder.GraphObject(0*honeycombCost), slgbuilder.GraphObject(0*honeycombCost), # no on-surface cost
                slgbuilder.GraphObject(self.helperWeight*backgroundCost), slgbuilder.GraphObject(self.helperWeight*backgroundCost), # extra 2 dark lines
                slgbuilder.GraphObject(self.helperWeight*1.5*(honeycombCost))] # Extra white line 
        if builderType == 'default':
            helper = slgbuilder.MaxflowBuilder(capacity_type=np.float64)
        elif builderType == 'parallel_2':
            helper = slgbuilder.PBKBuilder(capacity_type=np.float64, num_threads=2, expected_blocks=1)
        elif builderType == 'parallel_4':
            helper = slgbuilder.PBKBuilder(capacity_type=np.float64, num_threads=4, expected_blocks=1)
        elif builderType == 'parallel_6':
            helper = slgbuilder.PBKBuilder(capacity_type=np.float64, num_threads=6, expected_blocks=1)
        else:
            raise AttributeError("Wrong builder type, has to be either 'default' or 'parallel_{2,4,6}'")
        helper.add_objects(layers)

        ## Adding regional costs, 
        helper.add_layered_region_cost(layers[0], 255-honeycombCost, honeycombCost)
        helper.add_layered_region_cost(layers[1], honeycombCost, 255-honeycombCost)

        ## Adding geometric constrains
        # Blocks crossing from bottom to top of the image
        helper.add_layered_boundary_cost() 
        # Surface smoothness term
        helper.add_layered_smoothness(layers[0:2],delta=self.edgeSmoothness, wrap=False)
        helper.add_layered_smoothness(layers[2:5],delta=self.helperSmoothness, wrap=True)
        # Honeycomb edges pair  
        helper.add_layered_containment(layers[0], layers[1], min_margin=self.wallThickness[0], max_margin=self.wallThickness[1])
        # Dark helper surfaces 
        helper.add_layered_containment(layers[2], layers[3], min_margin=self.darkHelperDist[0], max_margin=self.darkHelperDist[1])
        # Top dark surface and top honeycomb edge 
        helper.add_layered_containment(layers[2], layers[0], min_margin=darkToHoneycombMinMargin) 
        # Bottom honeycomb edge and bottom dark surface
        helper.add_layered_containment(layers[1], layers[3], min_margin=darkToHoneycombMinMargin) 

        # Top dark surface and white central surface
        helper.add_layered_containment(layers[2], layers[4], min_margin=self.darkWhiteHelperDist[0], max_margin=self.darkWhiteHelperDist[1]) 
        # White central surface and bottom dark surface
        helper.add_layered_containment(layers[4], layers[3], min_margin=self.darkWhiteHelperDist[0], max_margin=self.darkWhiteHelperDist[1]) 

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
                    segm_plot, = plt.plot(segmentation_lines[i], 'r')
                else:
                    helper_plot, = plt.plot(segmentation_lines[i], 'b')

            plt.legend([segm_plot,helper_plot],["Detected honeycomb wall edges","detected helper surfaces"])
            plt.show()

        if return_helper_surfaces == False:
            segmentation_lines.pop(-1)
            segmentation_lines.pop(-1)
            segmentation_lines.pop(-1)
        
        return segmentation_lines

    
    def v1_detect(self, unfolded_img, honeycombCost, visualize=False):
        """ Performs the old version of detection of the honeycomb wall edges.\n
        Params:\n
        unfolded_img - clean unfolded image
        honeycombCost - cost function image for the honeycomb-targeting surfaces \n
        backgroundCost - cost function image for the background-targeting surfaces \n
        visualize - if True, plots the resulting surfaces on an unfolded image \n
        return_helper_surfaces - if True, returns also dark helper surfaces.
        """

        layers = [slgbuilder.GraphObject(0*honeycombCost), slgbuilder.GraphObject(0*honeycombCost)]
        helper = slgbuilder.MaxflowBuilder()
        helper.add_objects(layers)

        ## Adding regional costs, 
        helper.add_layered_region_cost(layers[0], 255-honeycombCost, honeycombCost)
        helper.add_layered_region_cost(layers[1], honeycombCost, 255-honeycombCost)

        ## Adding geometric constrains
        # Blocks crossing from bottom to top of the image
        helper.add_layered_boundary_cost() 
        # Surface smoothness term
        helper.add_layered_smoothness(layers[0:2],delta=self.edgeSmoothness, wrap=False)
        # Honeycomb edges pair  
        helper.add_layered_containment(layers[0], layers[1], min_margin=self.wallThickness[0], max_margin=self.wallThickness[1])

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
                plt.plot(segmentation_lines[i], 'r')
            plt.legend(["Detected honeycomb wall edges"])
            plt.show()

        
        return segmentation_lines


    def v2_detect(self, unfolded_img, honeycombCost, backgroundCost, visualize=False, return_helper_surfaces=False):
        """ Performs detection of the honeycomb wall edges.\n
        Params:\n
        unfolded_img - clean unfolded image
        honeycombCost - cost function image for the honeycomb-targeting surfaces \n
        backgroundCost - cost function image for the background-targeting surfaces \n
        visualize - if True, plots the resulting surfaces on an unfolded image \n
        return_helper_surfaces - if True, returns also dark helper surfaces.
        """

        darkToHoneycombMinMargin = 1 # min distance between the helper surface and honeycomb edge

        layers = [slgbuilder.GraphObject(0*honeycombCost), slgbuilder.GraphObject(0*honeycombCost), # no on-surface cost
                slgbuilder.GraphObject(self.helperWeight*backgroundCost), slgbuilder.GraphObject(self.helperWeight*backgroundCost)] # Extra white line 
        helper = slgbuilder.MaxflowBuilder(capacity_type=np.float64)
        helper.add_objects(layers)

        ## Adding regional costs, 
        helper.add_layered_region_cost(layers[0], 255-honeycombCost, honeycombCost)
        helper.add_layered_region_cost(layers[1], honeycombCost, 255-honeycombCost)

        ## Adding geometric constrains
        # Blocks crossing from bottom to top of the image
        helper.add_layered_boundary_cost() 
        # Surface smoothness term
        helper.add_layered_smoothness(layers[0:2],delta=self.edgeSmoothness, wrap=False)
        helper.add_layered_smoothness(layers[2:4],delta=self.helperSmoothness, wrap=True)
        # Honeycomb edges pair  
        helper.add_layered_containment(layers[0], layers[1], min_margin=self.wallThickness[0], max_margin=self.wallThickness[1])
        # Dark helper surfaces 
        helper.add_layered_containment(layers[2], layers[3], min_margin=self.darkHelperDist[0], max_margin=self.darkHelperDist[1])
        # Top dark surface and top honeycomb edge 
        helper.add_layered_containment(layers[2], layers[0], min_margin=darkToHoneycombMinMargin) 
        # Bottom honeycomb edge and bottom dark surface
        helper.add_layered_containment(layers[1], layers[3], min_margin=darkToHoneycombMinMargin) 

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