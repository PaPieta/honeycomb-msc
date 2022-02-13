# -*- coding: utf-8 -*-
"""
Created on Wed Feb 09 15:12:29 2022

@author: pawel
"""

import numpy as np
import matplotlib.pyplot as plt

SHELL_FILE_NAME = "testShell4Surf_v2.inp"



# Open raw segmentation surfaces
# surf_array = np.load('data/slicewise_z30_1Surf_raw.npy')
surf_array_obj = np.load('data/slicewise_z200_allSurf_raw.npy', allow_pickle=True)
# Retrieve list of surfaces from the object array
surf_array_list = []
for i in range(surf_array_obj.shape[0]):
    if i%2 == 0:
        surf_array = []
    surf = []
    for j in range(surf_array_obj.shape[1]):
        surfAxis = surf_array_obj[i,j]
        surf.append(surfAxis)
    surf_array.append(np.array(surf))
    if i%2 == 1:
        surf_array_list.append(np.array(surf_array))


# Open dummy inp file (for simplicity) and save to list of lines
with open("data/dummyShell4Surf.inp", "r") as f:
    contents = f.readlines()

for k in range(len(surf_array_list)):
    surf_array = surf_array_list[k]
    # Calculate  center position and thickness
    shell_array = np.empty((4,surf_array.shape[2],surf_array.shape[3]))
    for i in range(surf_array.shape[2]):
        for j in range(surf_array.shape[3]):
            shell_array[:3,i,j] = np.mean((surf_array[0,:,i,j],surf_array[1,:,i,j]), axis=0)
            shell_array[3,i,j] = np.linalg.norm(surf_array[0,:,i,j]-surf_array[1,:,i,j])

    # Plot thickness results
    # plt.figure()
    # plt.imshow(shell_array[3,:,:].transpose(), cmap='jet')
    # plt.colorbar()
    # plt.show()

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


    # Find line where nodes should go
    nodesIdx = contents.index(f"*Part, name=Part-{k+1}\n")+1
    # Write nodes backwards to counting avoid line index 
    for i in range(vertices.shape[0]-1,-1,-1):
        newLine = f"{vertices[i,0].astype('int')}, {round(vertices[i,1],2)}, {round(vertices[i,2],2)}, {round(vertices[i,3],2)}\n"
        contents.insert(nodesIdx+1, newLine)
    # Same with faces
    facesIdx = contents.index(f"*Nset, nset=BC-{k+1}-1, generate\n")-1
    for i in range(faces.shape[0]-1,-1,-1):
        newLine = f"{faces[i,0].astype('int')}, {faces[i,1].astype('int')}, {faces[i,2].astype('int')}, {faces[i,3].astype('int')}, {faces[i,4].astype('int')}\n"
        contents.insert(facesIdx+1, newLine)

    # Find and create 2 border sets
    nsetIdx = contents.index(f"*Nset, nset=BC-{k+1}-1, generate\n")
    newLine = f" 1, {int(X.shape[1])}, 1\n"
    contents.insert(nsetIdx+1, newLine)

    setStart = X.shape[1]*(X.shape[0]-1) + 1
    setEnd = X.shape[1]*(X.shape[0])
    nsetIdx = contents.index(f"*Nset, nset=BC-{k+1}-2, generate\n")
    newLine = f" {int(setStart)}, {int(setEnd)}, 1\n"
    contents.insert(nsetIdx+1, newLine)

    elsetIdx = contents.index(f"*Elset, elset=Set-{k+1}-3, generate\n")
    newLine = f" 1, {int(faces.shape[0])}, 1\n"
    contents.insert(elsetIdx+1, newLine)

    # Write thickness data
    # thicknessIdx = contents.index("*Nodal Thickness\n")
    thicknessIdx = elsetIdx+2
    for i in range(thicknesses.shape[0]-1,-1,-1):
        newLine = f"{thicknesses[i,0].astype('int')}, {round(thicknesses[i,1],2)}\n"
        contents.insert(thicknessIdx+1, newLine)
# Save the new file
with open(f"data/{SHELL_FILE_NAME}", "w+") as f:
    contents = "".join(contents)
    f.write(contents)