# -*- coding: utf-8 -*-
"""
Created on Wed Feb 09 15:12:29 2022

@author: pawel
"""

import numpy as np
import matplotlib.pyplot as plt

SHELL_FILE_NAME = "testShell.inp"



# Open raw segmentation surfaces
surf_array = np.load('data/slicewise_z30_1Surf_raw.npy')
# surf_array = np.load('data\slicewise_z200_allSurf_raw.npy')

# Calculate  center position and thickness
shell_array = np.empty((4,surf_array.shape[2],surf_array.shape[3]))
for i in range(surf_array.shape[2]):
    for j in range(surf_array.shape[3]):
        shell_array[:3,i,j] = np.mean((surf_array[0,:,i,j],surf_array[1,:,i,j]), axis=0)
        shell_array[3,i,j] = np.linalg.norm(surf_array[0,:,i,j]-surf_array[1,:,i,j])

# Plot thickness results
plt.figure()
plt.imshow(shell_array[3,:,:].transpose(), cmap='jet')
plt.colorbar()
plt.show()

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

# Open dummy inp file (for simplicity) and save to list of lines
with open("data/dummyShell.inp", "r") as f:
    contents = f.readlines()
# Find line where nodes should go
nodesIdx = contents.index("*Node\n")
# Write nodes backwards to counting avoid line index 
for i in range(vertices.shape[0]-1,-1,-1):
    newLine = f"{vertices[i,0].astype('int')}, {round(vertices[i,1],2)}, {round(vertices[i,2],2)}, {round(vertices[i,3],2)}\n"
    contents.insert(nodesIdx+1, newLine)
# Same with faces
facesIdx = contents.index("*Element, type=S4R\n")
for i in range(faces.shape[0]-1,-1,-1):
    newLine = f"{faces[i,0].astype('int')}, {faces[i,1].astype('int')}, {faces[i,2].astype('int')}, {faces[i,3].astype('int')}, {faces[i,4].astype('int')}\n"
    contents.insert(facesIdx+1, newLine)

# Find and create 2 border sets
nsetIdx = contents.index("*Nset, nset=Set-1, generate\n")
newLine = f" 1, {int(X.shape[1])}, 1\n"
contents.insert(nsetIdx+1, newLine)

setStart = X.shape[1]*(X.shape[0]-1) + 1
setEnd = X.shape[1]*(X.shape[0])
nsetIdx = contents.index("*Nset, nset=Set-2, generate\n")
newLine = f" {int(setStart)}, {int(setEnd)}, 1\n"
contents.insert(nsetIdx+1, newLine)

# elsetIdx = contents.index("*Elset, elset=Set-1, generate\n")
# newLine = f" 1, {int(faces.shape[0])}, 1\n"
# contents.insert(elsetIdx+1, newLine)
# Write thickness data
thicknessIdx = contents.index("*Nodal Thickness\n")
for i in range(thicknesses.shape[0]-1,-1,-1):
    newLine = f"{thicknesses[i,0].astype('int')}, {round(thicknesses[i,1],2)}\n"
    contents.insert(thicknessIdx+1, newLine)
# Save the new file
with open(f"data/{SHELL_FILE_NAME}", "w+") as f:
    contents = "".join(contents)
    f.write(contents)