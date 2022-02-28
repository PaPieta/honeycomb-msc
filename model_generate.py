# -*- coding: utf-8 -*-
"""
Created on Wed Feb 09 15:12:29 2022

@author: pawel
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.ndimage as ndimage
import scipy.interpolate
import helpers
import vtk_write_lite as vwl


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


def surf_to_shell_simple(surf2_array):
    """Calculates a shell central surface (consisting of 3D position and thickness values), from array with 2 surfaces.\n
    Params:\n
    surf2_array - array with 2 surfaces, dim: (2,3,x,z)
    """
    # Calculate  center position and thickness
    shell_array = np.empty((4,surf2_array.shape[2],surf2_array.shape[3]))
    for i in range(surf2_array.shape[2]):
        for j in range(surf2_array.shape[3]):
            shell_array[:3,i,j] = np.mean((surf2_array[0,:,i,j],surf2_array[1,:,i,j]), axis=0)
            shell_array[3,i,j] = np.linalg.norm(surf2_array[0,:,i,j]-surf2_array[1,:,i,j])

    return shell_array


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
    ndotu = np.sum(surf1_normals*surf2_normals,axis=0) #dot product for each matrix element
    if np.any(abs(ndotu)<epsilon):
        raise RuntimeError("no intersection or line is within plane for one of the points")
    w = surf2 - surf1
    si = -np.sum(surf1_normals*w,axis=0)/ndotu
    Psi = w + si * surf2_normals + surf1
    return Psi


def surf_to_shell_interp(surf2_array,step=[3,3],sigma=2,pixel_size=1,return_surf=False):
    """Performs smoothing and interpolation of the provided surface pair. Next, calculates the shell array 
    by finding a mean of the surfaces and thickness using surface normals. 
    Returns the shell surface and the original interpolated surface pair (if requested)\n
    Params:\n
    surf2_array - array with 2 surfaces, dim: (2,3,x,z)\n
    step - 2 element array with step in x and z direction (in pixels)\n
    sigma - smoothing coefficient (in pixels)\n
    pixel_size - multiplier for correct data representation\n
    return_surf - if True returns shell and interpolated surface pair in a list, if False returns only shell
    """
    # Smooth surfaces
    surf2_array[0,1,:,:] = ndimage.gaussian_filter(surf2_array[0,1,:,:], sigma=(sigma, sigma), order=0)
    surf2_array[1,1,:,:] = ndimage.gaussian_filter(surf2_array[1,1,:,:], sigma=(sigma, sigma), order=0)

    xPoints = int((np.max(surf2_array[:,0,:,:])-np.min(surf2_array[:,0,:,:]))/step[0])
    zPoints = int((np.max(surf2_array[:,2,:,:])-np.min(surf2_array[:,2,:,:]))/step[1])

    surf_array_1 = helpers.getSurfUniformSpacing(surf2_array[0,:,:,:], zPoints=zPoints, xPoints=xPoints)
    surf_array_2 = helpers.getSurfUniformSpacing(surf2_array[1,:,:,:], zPoints=zPoints, xPoints=xPoints)
    
    surf2_array = np.array([surf_array_1,surf_array_2])

    # Calculate central surface
    center_surf_array = np.mean(surf2_array,axis=0)

    surfNormalsMat1 = surf_3d_normals(surf2_array[0,:,:,:])
    surfNormalsMat2 = surf_3d_normals(surf2_array[1,:,:,:])
    centerNormalsMat = surf_3d_normals(center_surf_array)

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
        return [surf2_array, shell_array]
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


def fem_model_shell_list(shell_array_list, part_file_path, master_file_path, save_path):
    """Creates finite element model from list of shell structures. Generates a master file and calls for generation of the part files.\n
    Params:\n
    shell_array_list - list of shell arrray models
    part_file_path - path to a template part file
    master_file_path - path to a template master file
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


if __name__ == "__main__":
    # Open raw segmentation surfaces
    # surf_array = np.load('data/slicewise_z30_1Surf_raw.npy')
    # surf_array_obj = np.load('data/slicewise_z200New_allSurf_raw.npy', allow_pickle=True)
    # surf_array_obj = np.load('data/slicewise_z250-750_allSurf_raw3.npy', allow_pickle=True)
    surf_array_obj = np.load('data/slicewise_z200-780_allSurf_raw.npy', allow_pickle=True)
    pixelSize = 0.0078329 #mm

    # Retrieve list of surfaces from the object array
    surf_array_list = npy_to_surf_list(surf_array_obj)

    shell_array_list = []
    plt.figure()
    for i in range(int(len(surf_array_list)/2)):
        surf2_array = np.array([surf_array_list[i*2],surf_array_list[(i*2)+1]])
        surf2_array = surf2_array[:,:,:,70:]
        [surf2_array, shell_array] = surf_to_shell_interp(surf2_array,step=[3,3],sigma=2,pixel_size=pixelSize,return_surf=True)
        # shell_array = surf_to_shell_simple(surf2_array)
        shell_array_list.append(shell_array)
        surf_array_list[i*2] = surf2_array[0,:,:,:]
        surf_array_list[(i*2)+1] = surf2_array[1,:,:,:]

        # Visualization
        plt.subplot(2,2,i+1)
        plt.imshow(shell_array[3,:,:].transpose(), cmap='jet')
        plt.colorbar()
    plt.show()

    # vwl.save_multSurf2vtk('data/surfaces/slicewise_z200-780_12_resized.vtk', surf_array_list)
    # vwl.save_multSurf2vtk('data/surfaces/slicewise_z200-780_12_center_resized.vtk', shell_array_list)

    # partFilePath = "data/abaqusShells/dummyPart.inp"
    # masterFilePath = "data/abaqusShells/dummyMaster.inp"
    # savePath = "data/abaqusShells/H29_slicewise_z200-780_4x/"

    # fem_model_shell_list(shell_array_list, partFilePath, masterFilePath, savePath)