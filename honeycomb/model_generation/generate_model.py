# -*- coding: utf-8 -*-
"""
Created on Wed Feb 09 15:12:29 2022

@author: Pawel Pieta s202606@student.dtu.dk
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.ndimage as ndimage
import scipy.interpolate
import scipy.linalg

from honeycomb.helpers import misc
from honeycomb.helpers import vtk_write_lite as vwl


def regularization_matrix(N, alpha, beta):
    """An NxN matrix for imposing elasticity and rigidity to snakes.
    Arguments: alpha is weigth for second derivative (elasticity),
    beta is weigth for (-)fourth derivative (rigidity)."""
    column = np.zeros(N)
    column[[-2,-1,0,1,2]] = alpha*np.array([0,1,-2,1,0]) + beta*np.array([-1,4,-6,4,-1])
    A = scipy.linalg.toeplitz(column)
    return(scipy.linalg.inv(np.eye(N)-A))

def npy_to_surf_list(surf_array_obj):
    """Converts the numpy "object array" containing segmeted surfaces to a list of surfaces.\n
    Params:\n
    surf_array_obj - object array with surfaces, usually loaded from npy file.
    """

    surf_array_list = []
    for i in range(surf_array_obj.shape[0]):
        surf_array = []
        for j in range(surf_array_obj.shape[1]):
            surfAxis = surf_array_obj[i,j]
            surf_array.append(surfAxis)
        surf_array_list.append(np.array(surf_array))

    return surf_array_list

def surf_3d_normals(surf_array):
    """Calculates a normals matrix from a surface. The created lines are normal to the surface defined by the neighbours of each point.\n
    Params:\n
    surf_array - surfae array, dim: (3,x,z)
    """
    pointsArray = np.zeros((3,surf_array.shape[0],surf_array.shape[1],surf_array.shape[2]))
    #Bottom left triangle point
    pointsArray[0,:,1:,1:] = surf_array[:,:-1,:-1]
    pointsArray[0,:,0,1:] = surf_array[:,0,:-1]
    pointsArray[0,:,1:,0] = surf_array[:,:-1,0]
    pointsArray[0,:,0,0] = surf_array[:,0,0]
    #Bottom right triangle point
    pointsArray[1,:,:-1,1:] = surf_array[:,1:,:-1]
    pointsArray[1,:,-1,1:] = surf_array[:,-1,:-1]
    pointsArray[1,:,:-1,0] = surf_array[:,1:,0]
    pointsArray[1,:,-1,0] = surf_array[:,-1,0]
    #Top triangle point
    pointsArray[2,:,:,:-1] = surf_array[:,:,1:]
    pointsArray[2,:,:,-1] = surf_array[:,:,-1]
    #Calculate vectors for cross product
    trVec1 = pointsArray[1,:,:] - pointsArray[0,:,:]
    trVec2 = pointsArray[2,:,:] - pointsArray[0,:,:]
    #Calculate normal vector
    normalsMat = np.cross(trVec1,trVec2, axis=0)
    #Normalize
    normalsMat = normalsMat/np.apply_along_axis(np.linalg.norm,0,normalsMat)

    return normalsMat

def surf_normals_surf_intersect(surf1, surf1_normals, surf2, surf2_normals, epsilon=1e-6):
    """Calcualtes intersection points between a surface and rays defined by normal vectors of other surface\n
    Params:\n
    surf1, surf1_normals - surface and its normal vector - on this suface the intersection will be found\n
    surf2, surf2_normals - surface and its normal vector - this surface will be treated as the ray start
    """
    # Source: https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python
    ndotu = np.sum(surf1_normals*surf2_normals,axis=0) #dot product for each matrix element
    if np.any(abs(ndotu)<epsilon):
        raise RuntimeError("no intersection or line is within plane for one of the points")
    w = surf2 - surf1
    si = -np.sum(surf1_normals*w,axis=0)/ndotu
    Psi = w + si * surf2_normals + surf1
    return Psi

def surf_to_shell_interp(surf2_array,step=[3,3],sigma=2,pixel_size=1,return_surf=False):
    """Performs smoothing and initial interpolation of the provided surface pair. Next, calculates the central surface 
    by finding a mean of the surfaces and vectors normal to all thre surfaces. Finally, resamples the normals and surfaces
    to a desired step and finds the thickness using the normals.
    Returns the shell surface and the original interpolated surface pair (if requested)\n
    Params:\n
    surf2_array - array with 2 surfaces, dim: (2,3,x,z)\n
    step - 2 element array with step in x and z direction (in pixels)\n
    sigma - smoothing coefficient (in pixels)\n
    pixel_size - multiplier for correct data representation\n
    return_surf - if True returns shell and interpolated surface pair in a list, if False returns only shell
    """
    # Smooth surfaces
    surf2_array[0,:2,:,:] = ndimage.gaussian_filter(surf2_array[0,:2,:,:], sigma=(0, sigma, sigma), order=0)
    surf2_array[1,:2,:,:] = ndimage.gaussian_filter(surf2_array[1,:2,:,:], sigma=(0, sigma, sigma), order=0)

    # Initial interpolation to fix uneven mesh
    surf2_array = misc.getMultSurfUniformSpacing(surf2_array, zStep=(np.max(surf2_array[:,2,:,:])+1)/surf2_array.shape[3], xStep=(np.max(surf2_array[:,0,:,:])+1)/surf2_array.shape[2])

    # Calculate surface normals first
    surfNormalsMat1 = surf_3d_normals(surf2_array[0,:,:,:])
    surfNormalsMat2 = surf_3d_normals(surf2_array[1,:,:,:])

    # Calculate central surface and its normals
    center_surf_array = np.mean(surf2_array,axis=0)
    centerNormalsMat = surf_3d_normals(center_surf_array)

    # Interpolate surfaces and normals
    surf3_normals = np.array([surfNormalsMat1,surfNormalsMat2,centerNormalsMat])
    [surf3_array, surf3_normals] = misc.getMultSurfUniformSpacing(np.array([surf2_array[0,:,:,:],surf2_array[1,:,:,:],center_surf_array]), zStep=step[1], xStep=step[0], mult_normals=surf3_normals)
    surfNormalsMat1 = surf3_normals[0,:,:,:]
    surfNormalsMat2 = surf3_normals[1,:,:,:]
    centerNormalsMat = surf3_normals[2,:,:,:]
    surf2_array = surf3_array[0:2,:,:,:]
    center_surf_array = surf3_array[2,:,:,:]

    # Calculate intersection
    surfIntersect1 = surf_normals_surf_intersect(surf2_array[0,:,:,:],surfNormalsMat1,center_surf_array,centerNormalsMat)
    surfIntersect2 = surf_normals_surf_intersect(surf2_array[1,:,:,:],surfNormalsMat2,center_surf_array,centerNormalsMat)

    # Calculate thickness
    thickness_surf_array = np.apply_along_axis(np.linalg.norm,0,surfIntersect1-surfIntersect2)

    # Combine results to shell
    shell_array = np.concatenate((center_surf_array,np.array([thickness_surf_array])), axis=0)

    # Resize
    surf2_array = surf2_array*pixel_size
    shell_array = shell_array*pixel_size

    if return_surf == True:
        return [shell_array, surf2_array]
    else:
        return shell_array

def fem_model_shell(shell_array, part_file_path, save_path, shell_idx=1):
    """Creates finite element model of a part from a shell surface.\n
    Params:\n
    shell_array - array defining the shell model, dim: (4,x,z)\n
    part_file_path - path to the template part file\n
    save_path - path where to save the created part file\n
    shell_idx - index of a shell in the whole model.
    """

    with open(part_file_path, "r") as f:
        contents = f.readlines()

    # Prepare the data for saving in .inp file
    X = shell_array[0].transpose()
    Y = shell_array[1].transpose()
    Z = shell_array[2].transpose()
    thic = shell_array[3].transpose()

    indices = np.arange(X.size).reshape(X.shape) + 1
    vertices = np.c_[indices.ravel(), X.ravel(), Y.ravel(), Z.ravel()]
    thicknesses = np.c_[indices.ravel(), thic.ravel()]

    lu = indices[:-1,:-1]
    ru = indices[:-1,1:]
    rb = indices[1:,1:]
    lb = indices[1:,:-1]
    faces = np.c_[np.arange(lu.size)+1, lu.ravel(), ru.ravel(), rb.ravel(), lb.ravel()]

    #Find and rename part header
    partIdx = contents.index("*Part, name=PART-X\n")
    contents[partIdx] = f"*Part, name=PART-{shell_idx}\n"
    # Find line where nodes should go
    nodesIdx = partIdx+1
    # Write nodes backwards to counting avoid line index 
    for i in range(vertices.shape[0]-1,-1,-1):
        newLine = f"{vertices[i,0].astype('int')}, {round(vertices[i,1],4)}, {round(vertices[i,2],4)}, {round(vertices[i,3],4)}\n"
        contents.insert(nodesIdx+1, newLine)
    # Same with faces
    facesIdx = contents.index(f"*Element, type=S4R\n")
    for i in range(faces.shape[0]-1,-1,-1):
        newLine = f"{faces[i,0].astype('int')}, {faces[i,1].astype('int')}, {faces[i,2].astype('int')}, {faces[i,3].astype('int')}, {faces[i,4].astype('int')}\n"
        contents.insert(facesIdx+1, newLine)
    # Write thickness data
    thicknessIdx = contents.index("*Nodal Thickness\n")
    for i in range(thicknesses.shape[0]-1,-1,-1):
        newLine = f"{thicknesses[i,0].astype('int')}, {round(thicknesses[i,1],4)}\n"
        contents.insert(thicknessIdx+1, newLine)

    # Find and create 2 horizontal border sets
    nsetIdx = contents.index("*Nset, nset=BCdown, generate\n")
    newLine = f" 1, {int(X.shape[1])}, 1\n"
    contents.insert(nsetIdx+1, newLine)

    setStart = X.shape[1]*(X.shape[0]-1) + 1
    setEnd = X.shape[1]*(X.shape[0])
    nsetIdx = contents.index("*Nset, nset=BCtop, generate\n")
    newLine = f" {int(setStart)}, {int(setEnd)}, 1\n"
    contents.insert(nsetIdx+1, newLine)

    # Find and create 2 vertical border sets
    setStart = 1
    setEnd = X.shape[1]*(X.shape[0]-1) + 1
    setStep = X.shape[1]
    nsetIdx = contents.index("*Nset, nset=BCV0, generate\n")
    newLine = f" {int(setStart)}, {int(setEnd)}, {int(setStep)}\n"
    contents.insert(nsetIdx+1, newLine)

    setStart = X.shape[1]
    setEnd = X.shape[1]*(X.shape[0])
    setStep = X.shape[1]
    nsetIdx = contents.index("*Nset, nset=BCV1, generate\n")
    newLine = f" {int(setStart)}, {int(setEnd)}, {int(setStep)}\n"
    contents.insert(nsetIdx+1, newLine)

    # Create all element set
    elsetIdx = contents.index(f"*Elset, elset=Model, generate\n")
    # contents[elsetIdx] = f"*Elset, elset=Model-{shell_idx}, generate\n"
    newLine = f" 1, {int(faces.shape[0])}, 1\n"
    contents.insert(elsetIdx+1, newLine)

    # Save the new part file
    fileName = f"part{shell_idx}.inp"
    fullSavePath = save_path + fileName
    with open(fullSavePath, "w+") as f:
        contents = "".join(contents)
        f.write(contents)

    return fileName


def save_fem_shell_model(shell_array_list, part_file_path, master_file_path, save_path):
    """Creates finite element model from list of shell structures. Generates a master file and calls for generation of the part files.\n
    Params:\n
    shell_array_list - list of shell arrray models\n
    part_file_path - path to a template part file\n
    master_file_path - path to a template master file\n
    save_path - path to a new folder where the model files will be placed
    """

    # Open master file and get contents
    with open(master_file_path, "r") as f:
        contents = f.readlines()
    # Change model name
    modelName = os.path.basename(os.path.normpath(save_path))
    nameIdx = contents.index("** Model name: XXX\n")
    contents[nameIdx] = f"** Model name: {modelName}\n"

    # Create thefolder iff doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    partIdx = contents.index("** PARTS\n")+2
    #Loop through shells and generate part files
    for i in range(len(shell_array_list)):
        shell_array = shell_array_list[i]

        partFileName = fem_model_shell(shell_array, part_file_path, save_path, shell_idx=i+1)
        # Add created part to the master file
        newLine = f"*INCLUDE, input={partFileName}\n"
        contents.insert(partIdx, newLine)
        partIdx = partIdx + 1

    # Save the new master file
    fileName = f"{modelName}.inp"
    fullSavePath = save_path + fileName
    with open(fullSavePath, "w+") as f:
        contents = "".join(contents)
        f.write(contents)


def generate_shell_model(surf_array_list,pixelSize=1,meshSize=[6,6],smoothingSigma=1,returnSurfaces=True, plotThickness=True):
    """Generates a shell representation of the honeycomb structure from its segmented wall edges.
    Params:\n
    surf_array - list of surface arrays representing the 3D segmentation result of the honeycomb wall edges\n
    pixel_size - used to rescale the model size to fit the original representation\n
    meshSize - [X,Z] axis size of the generated mesh
    smoothingSigma - sigma parameter in the gaussian smoothing of the surface\n
    returnSurfaces - if True, returns also original surfaces (reinterpolated and smoothed)\n
    plotThickness - if True, plots the calculated wall thickness data
    """
    # Calculate shell surfaces
    shell_array_list = []
    minList = []
    maxList = []
    for i in range(int(len(surf_array_list)/2)):
        surf2_array = np.array([surf_array_list[i*2],surf_array_list[(i*2)+1]])
        [shell_array, surf2_array] = surf_to_shell_interp(surf2_array,step=meshSize,sigma=smoothingSigma,pixel_size=pixelSize,return_surf=True)
        shell_array_list.append(shell_array)
        surf_array_list[i*2] = surf2_array[0,:,:,:]
        surf_array_list[(i*2)+1] = surf2_array[1,:,:,:]

        minList.append(np.min(shell_array[3,:,:]))
        maxList.append(np.max(shell_array[3,:,:]))

    if plotThickness:
        if len(shell_array_list) == 4:
            fig, ax = plt.subplots(2,2,gridspec_kw={'width_ratios': [1, 2.2]})
            posLookup = [0,1,3,2]
        else:
            fig, ax = plt.subplots(4,2)
            posLookup = [0,1,2,3,4,5,6,7]
            # posLookup = [0,2,4,6,7,5,3,1]
        minVal = np.min(np.array(minList))
        maxVal = np.max(np.array(maxList))
        for i in range(len(shell_array_list)):
            im = ax.flat[posLookup[i]].imshow(shell_array_list[i][3,:,:].transpose(), vmin=minVal,vmax=maxVal, cmap='jet',extent=[0,meshSize[0]*pixelSize*shell_array_list[i][0,:,:].shape[0],0,meshSize[1]*pixelSize*shell_array_list[i][0,:,:].shape[1]])
            ax.flat[posLookup[i]].set_title(f'Wall  {i+1}',fontweight="bold")
            ax.flat[posLookup[i]].set_xlabel("Width [mm]")
            ax.flat[posLookup[i]].set_ylabel("Height [mm]")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Wall thickness [mm]', rotation=270, labelpad=10)
        plt.show()

    if returnSurfaces:
        return [shell_array_list, surf_array_list]
    else:
        return shell_array_list

if __name__ == "__main__":
    # Open raw segmentation surfaces
    # surf_array_obj = np.load('data/rawFinal/slicewise_z200-780_allSurf_raw.npy', allow_pickle=True)
    # surf_array_obj = np.load('data/H29_slicewise_z200-780_allSurf_raw.npy', allow_pickle=True)
    surf_array_obj = np.load('data/rawFinal/H29big_slicewise_z380-620_allSurf_raw.npy', allow_pickle=True)
    # surf_array_obj = np.load('data/H29big_slicewise_z380-620_allSurf_raw.npy', allow_pickle=True)
    # surf_array_obj = np.load('data/rawFinal/NLbig_slicewise_z390-640_allSurf_raw_2.npy', allow_pickle=True)
    # surf_array_obj = np.load('data/NLbig_slicewise_z390-640_rot_-15_allSurf_raw.npy', allow_pickle=True)
    # surf_array_obj = np.load('data/rawFinal/NL_slicewise_z340-790_allSurf_raw_2.npy', allow_pickle=True)
    # surf_array_obj = np.load('data/NL_slicewise_z340-790_rot_-15_allSurf_raw.npy', allow_pickle=True)
    # surf_array_obj = np.load('data/rawFinal/PD_slicewise_z0-950_allSurf_raw.npy', allow_pickle=True)
    # surf_array_obj = np.load('data/rawFinal/PBbig_slicewise_z250-780_allSurf_raw.npy', allow_pickle=True)
    # pixelSize = 0.0078329 #mm 
    pixelSize = 0.017551 #mm 
    # pixelSize = 0.031504 #mm 
    # pixelSize = 0.015172 #mm 
    # pixelSize = 0.015752 #mm 
    # pixelSize = 0.032761 #mm 
    # pixelSize = 1 # For ParaView

    meshSize = [6,6]

    # Retrieve list of surfaces from the object array
    surf_array_list = npy_to_surf_list(surf_array_obj)

    # Generate shell model
    [shell_array_list, surf_array_list] = generate_shell_model(surf_array_list,
                                                                pixelSize=pixelSize,
                                                                meshSize=meshSize,
                                                                smoothingSigma=1,
                                                                returnSurfaces=True,
                                                                plotThickness=True)

    #Saving the data
    # vwl.save_multSurf2vtk('data/surfaces/H29_slicewise_z200-780_1px.vtk', surf_array_list)
    # vwl.save_multSurf2vtk('data/surfaces/H29_slicewise_z200-780_center_coloured.vtk', shell_array_list)

    # partFilePath = "data/abaqusShells/dummyPart.inp"
    # masterFilePath = "data/abaqusShells/dummyMaster.inp"
    # savePath = "data/abaqusShells/H29_slicewise_z200-780_4x_meanThickness/"

    # save_fem_shell_model(shell_array_list, partFilePath, masterFilePath, savePath)