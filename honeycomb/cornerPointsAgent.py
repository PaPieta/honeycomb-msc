# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:01:13 2022

@author: Pawel Pieta s202606@student.dtu.dk
"""

import numpy as np 
import os
import skimage.io 
import scipy.ndimage as scp

from honeycomb.unfolding.unfold3d import Unfold3d
from honeycomb.unfolding.unfold2d import Unfold2d


def save2dHcPoints(savePath, hcList):
    """Saves the the points marked by the user in a 2d honeycomb image into a txt file\n
    Params:\n
    savePath - Full file save path
    """

    saveDir = os.path.dirname(savePath)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    with open(savePath, 'w') as f:
        for i in range(len(hcList)):
            hc = hcList[i]
            line = hc.lines
            f.write(f"HC {i} {line.shape[1]}\n")
            for k in range(line.shape[1]):
                f.write(f"{line[0,k]} {line[1,k]}\n")

def load2dHcPoints(loadPath, hcList):
    """Loads from txt file ponits previously marked by the user in a 2d honeycomb image.\n
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
            if prevHcIdx != -1:
                hcList[prevHcIdx].lines = line

            numPoints =  int(elemList[2])
            pointsCounter = 0
            line = np.zeros((2,numPoints))
        elif currString != '\n':
            line[:,pointsCounter] = np.array([float(elemList[0]), float(elemList[1])])
            pointsCounter += 1
    
    hcList[hcIdx].lines = line

    return hcList


def save3dHcPoints(savePath, hcList):
    """ Saves the the points marked by the user in a volumetric honeycomb image into a txt file\n
    Params:\n
    savePath - Full file save path
    """

    saveDir = os.path.dirname(savePath)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    with open(savePath, 'w') as f:
        for i in range(len(hcList)):
            hc = hcList[i]
            for j in range(len(hc.lines)):
                line = hc.lines[j]
                f.write(f"HC {i} {j} {line.shape[1]}\n")
                for k in range(line.shape[1]):
                    f.write(f"{line[0,k]} {line[1,k]} {line[2,k]}\n")

def load3dHcPoints(loadPath, hcList):
    """Loads from txt file ponits previously marked by the user in a volumetric honeycomb image.\n
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
                    hcList[prevHcIdx].lines = lineList
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
    hcList[hcIdx].lines = lineList

    return hcList


if __name__ == "__main__":

    # Load a chosen dataset
    I = skimage.io.imread('data/NL07C_NL07C-60kV-zoom-center_recon.tif')



    ### 2D version ###

    # Choose slice and save name
    I_slice = I[240,:,:]
    # Rotate if needed
    I_slice = scp.rotate(I_slice,-15)
    savePointsPath = "data/cornerPoints/NL_z240_1wall_highNumPoints.txt"
    # Choose number of walls to mark
    wallNum = 1
    # Preparing separate unfolding objects for each wall
    visImg = np.copy(I_slice)
    hcList = [Unfold2d(I_slice, visImg, visualize=False) for i in range(wallNum)]
    # Draw corners
    for hc in hcList:
        visImg = hc.draw_corners()
    # Save the corners
    save2dHcPoints(savePointsPath, hcList)



    ### 3D version ###

    # I_cut= I[200:400,:,:]
    # savePointsPath = "data/cornerPoints/H29_z200-400_test.txt"
    # # Choose number of walls to mark
    # wallNum = 4
    # # Preparing separate unfolding objects for each wall
    # visImg = np.copy(I_cut)
    # hcList = [Unfold3d(I_cut, visImg, visualize=False) for i in range(wallNum)]
    # # Draw corners
    # for hc in hcList:
    #     visImg = hc.draw_corners()
    # # Save the corners
    # save3dHcPoints(savePointsPath, hcList)