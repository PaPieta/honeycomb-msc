#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:06:05 2019

@author: vand
@modifications: Pawel Pieta s202606@student.dtu.dk
"""

import numpy as np
import matplotlib.pyplot as plt
    
def save_gray2vtk(volume, filename, filetype='ASCII', origin=(0,0,0),
                  spacing=(1,1,1), dataname='gray'):
    ''' Writes a vtk file with grayscace volume data.
    Arguments:
       volume: a grayscale volume, values will be saved as floats
       filename: filename with .vtk extension
       filetype: file type 'ASCII' or 'BINARY'. Writing a binary file might not
           work for all OS due to big/little endian issues.
       origin: volume origin, defaluls to (0,0,0)
       spacing: volume spacing, defaults to 1
       dataname: name associated with data (will be visible in Paraview)
    Author:vand@dtu.dk, 2019
    '''
    with open(filename, 'w') as f:
        # writing header
        f.write('# vtk DataFile Version 3.0\n')
        f.write('saved from python using save_gray2vtk\n')
        f.write('{}\n'.format(filetype))
        f.write('DATASET STRUCTURED_POINTS\n')
        f.write('DIMENSIONS {} {} {}\n'.format(\
                volume.shape[2],volume.shape[1],volume.shape[0]))
        f.write('ORIGIN {} {} {}\n'.format(origin[0],origin[1],origin[2]))
        f.write('SPACING {} {} {}\n'.format(spacing[0],spacing[1],spacing[2]))
        f.write('POINT_DATA {}\n'.format(volume.size))
        f.write('SCALARS {} float 1\n'.format(dataname))
        f.write('LOOKUP_TABLE default\n')
        
    # writing volume data
    if filetype.upper()=='BINARY':
        with open(filename, 'ab') as f:
            volume = volume.astype('float32') # Pareview expects 4-bytes float 
            volume.byteswap(True) # Paraview expects big-endian 
            volume.tofile(f)
    else: # ASCII
        with open(filename, 'a') as f:
            np.savetxt(f,volume.ravel(),fmt='%.5g', newline= ' ')
    
def save_surf2vtk(filename, X,Y,Z):
     '''  Writes a vtk file for a 3D surface
     '''
     vertices = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
     indices = np.arange(X.size).reshape(X.shape)
     lu = indices[:-1,:-1]
     ru = indices[:-1,1:]
     rb = indices[1:,1:]
     lb = indices[1:,:-1]
     faces = np.c_[4*np.ones(lu.size), lu.ravel(), ru.ravel(), rb.ravel(), lb.ravel()]
     nf = faces.shape[0]
     
     with open(filename, 'w') as f:
         f.write('# vtk DataFile Version 3.0\n')
         f.write('saved from matlab using save_surf2vtk\n')
         f.write('ASCII\n')
         f.write('DATASET POLYDATA\n')
         f.write('POINTS {} float\n'.format(X.size))
         np.savetxt(f, vertices, fmt='%.5g', newline='\n')
         f.write('POLYGONS {} {}\n'.format(nf,5*nf))
         np.savetxt(f, faces, fmt='%d', newline='\n')

def save_multSurf2vtk(filename, surfacesList):
    '''  Writes a vtk file for multiple 3D surfaces
    '''
    # Check if color information should be added
    if surfacesList[0].shape[0]>3:
        writeColorData = True
        minList = []
        maxList = []
        for i in range(len(surfacesList)):
            minList.append(np.min(surfacesList[i][3]))
            maxList.append(np.max(surfacesList[i][3]))
        minVal = np.min(np.array(minList))
        maxVal = np.max(np.array(maxList))
        norm = plt.Normalize(minVal, maxVal)

        colorsFull = np.empty((0,3))
    else:
        writeColorData = False

    verticesFull = np.empty((0,3))
    facesFull = np.empty((0,5))
    for i in range(len(surfacesList)):
        X = surfacesList[i][0]
        Y = surfacesList[i][1]
        Z = surfacesList[i][2]

        vertices = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
        indices = np.arange(X.size).reshape(X.shape)
        lu = indices[:-1,:-1]
        ru = indices[:-1,1:]
        rb = indices[1:,1:]
        lb = indices[1:,:-1]
        faces = np.c_[4*np.ones(lu.size), lu.ravel(), ru.ravel(), rb.ravel(), lb.ravel()]
        faces[:,1:] = faces[:,1:] + verticesFull.shape[0]

        verticesFull = np.vstack((verticesFull,vertices))
        facesFull = np.vstack((facesFull,faces))

        if writeColorData:
            colorData = surfacesList[i][3]
            # Get color in jet colormap
            colors = plt.cm.jet(norm(colorData))
            # Remove alpha channel and flatten
            colors = colors[:,:,:3].reshape(-1,3)

            colorsFull = np.vstack((colorsFull,colors))

    nf = facesFull.shape[0]
     
    with open(filename, 'w') as f:
        f.write('# vtk DataFile Version 3.0\n')
        f.write('saved from matlab using save_multSurf2vtk\n')
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        f.write('POINTS {} float\n'.format(verticesFull.shape[0]))
        np.savetxt(f, verticesFull, fmt='%.5g', newline='\n')
        f.write('POLYGONS {} {}\n'.format(nf,5*nf))
        np.savetxt(f, facesFull, fmt='%d', newline='\n')

        if writeColorData:
            f.write('POINT_DATA {} \n'.format(verticesFull.shape[0]))
            f.write('COLOR_SCALARS label 3\n')
            np.savetxt(f, colorsFull, fmt='%.5g', newline='\n')
         



