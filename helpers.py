# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 16:45:10 2022

@author: pawel
"""

from random import gauss
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import RegularGridInterpolator
import scipy.interpolate

import os


def imgGaussianModel(img, step_3d=89):
    """Fits a gaussian mixture model (GMM) with 2 components to the image. 
    Works on a 3d stack, but takes every step_3d point.\n
    Params:\n
    img - image to fit GMM to\n
    step_3d - step with which the points are probed in the 3d img (use a prime number).
    """
    imgFlat = img.ravel()
    # If img is a 3d stack, take only a part of the pixels
    if img.ndim > 2:
        imgLen = imgFlat.shape[0]
        idxList = np.arange(0,imgLen,step_3d)
        imgFlat = np.take(imgFlat, idxList)

    # imgFlat = imgFlat[imgFlat > 1700]
    imgFlat = imgFlat[imgFlat > 20]

    gm = GaussianMixture(n_components=2, random_state=0)
    gm.fit(imgFlat.reshape(-1,1))
    means = gm.means_
    variances = np.array([[np.min(gm.covariances_)],[np.min(gm.covariances_)]])

    return means, variances

def gaussianProbFunction(img, mean, variance):
    """Calculates a cost image based on the unscaled pdf of provided gaussian.\n
    Params:\n
    img - image to calculate cost on\n
    mean - mean of the gaussians\n
    variance - variance of the gaussians (not used in this version).
    """
    img = np.abs(img - mean)
    cost = np.exp(-0.5*((img/np.sqrt(variance))**2.0))
    return cost

def sigmoidProbFunction(img, mean, variance, weight, visualize=False):
    """Calculates a cost image by fitting a sigmoid function
    between a pair of gaussians defined by a mean and variance.
    Slope of the sigmoid defines the class probability/cost.\n
    Params:\n
    img - image to calculate cost on\n
    mean - 2 element vector of means of the gaussians\n
    variance - 2 element vector of variances of the gaussians (not used in this version)\n
    weight - defines where should the center of the sigomid be placed. The bigger the weight, the closest it is to the first mean\n
    visualize - if True, will plot the fitted sigmoid and show the cost image.
    """
    #Sort means
    mean = np.sort(mean,axis=0)
    # This is the probability value at the mean of the gaussian
    gaussCenterProb = 0.99
    sigma = np.sqrt(variance)
    # scale = 1/(np.mean(sigma)/abs(mean[1]-mean[0]))

    #Center of the sigmoid in between the gaussians
    # center = np.mean(mean)
    center = mean[0]*weight + mean[1]*(1-weight)
    # Scale parameter calculated from the desired prob at gaussian mean 
    # scale = np.log(1/gaussCenterProb - 1)/(mean[0]-center)
    scale = np.log(1/gaussCenterProb - 1)/(mean[0]-np.mean(mean))
    # Sigmoid-based cost
    cost =  1/(1+np.exp(-scale*(img-center)))

    if visualize == True:
        
        x0 = np.linspace(mean[0] - 3*sigma[0], mean[0] + 3*sigma[0], 100)
        x1 = np.linspace(mean[1] - 3*sigma[1], mean[1] + 3*sigma[1], 100)
        xSigm = np.linspace(mean[0]-sigma[0], mean[1]+sigma[1], 300)

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.plot(x0, stats.norm.pdf(x0, mean[0], sigma[0]),color=color)
        ax1.plot(x1, stats.norm.pdf(x1, mean[1], sigma[1]),color=color)
        ax1.set_xlabel('Pixel value')
        ax1.set_ylabel('Gaussian distribution probability', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx() 

        color = 'tab:blue'
        ax2.plot(xSigm, 1/(1+np.exp(-scale*(xSigm-center))),color=color)
        ax2.set_ylabel('Pixel class probability', color=color)  
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.show()

    return cost

def surfaceNormal(poly):
    """Calculates a 3d vector normal to a surface defined by at least 3 points.\n
    Params:\n
    poly - list of 3d points defining a surface.
    """
    if len(poly) == 0:
        raise Exception('Define at least 3 points for surface normal vector calculation.')

    n = [0.0, 0.0, 0.0]

    for i, v_curr in enumerate(poly):
        v_next = poly[(i+1) % len(poly)]
        n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
        n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])

    normalised = [i/sum(n) for i in n]

    return normalised

def surfaceNormal2(poly):
    """Calculates a 3d vector normal to a surface defined by 3 points.\n
    Params:\n
    poly - array of 3d points defining a surface.
    """
    if poly.shape[0] != 3:
        raise Exception('Define 3 points for surface normal vector calculation.')

    U = poly[1,:] - poly[0,:]
    V = poly[2,:] - poly[0,:]
    N = np.cross(U,V)
    # Normalize
    N = N/np.linalg.norm(N)

    return N

def getFuncUniformSpacing(points, startX, endX, step=1, s=None, visualize=False):
    """Defines a uniform (in relation to data curve) x axis spacing 
    based on a 1st order spline fitted to the provided points.\n
    Params:\n
    points - 2d array with positions of the points\n
    startX - x axis value to start the interpolation from\n
    endX - x axis value to end the interpolation at.\n
    step - interpolation step\n
    s - Positive smoothing factor used to choose the number of knots. 
        Number of knots will be increased until the smoothing condition is satisfied.
        If 0, spline will interpolate through all data points\n
    visualize - if true, will plot resilt of the spline fitting and interpolation.
    """
    # Sort the points
    points = points[:,np.argsort(points[0,:])]
    # Make sure there are no points with same x val
    for i in range(points.shape[1]-1,-1,-1):
        if points[0,i] == points[0,i-1]:
            points[:,i-1] = np.mean((points[:,i],points[:,i-1]),axis=0)
            points = np.delete(points, i, axis=1)
    # # Sort again to be sure 
    # points = points[:,np.argsort(points[0,:])]
    # Calculate spline and get example values
    spl = UnivariateSpline(points[0,:], points[1,:], k=1, s=s)
    # x = np.arange(np.min(points[0,:]), np.max(points[0,:]), step)
    x = np.arange(startX, endX, step)
    y = spl(x)
    # https://stackoverflow.com/questions/19117660/how-to-generate-equispaced-interpolating-values
    #  Calculate all all distances between the points 
    # and generate the coordinates on the curve by cumulative summing.
    xd = np.diff(x)
    yd = np.diff(y)
    dist = np.sqrt(xd**2+yd**2)
    u = np.cumsum(dist)
    u = np.hstack([[0],u])
    # Interpolate the x-coordinates independently with respect to the new coordinates.
    t = np.arange(0,u.max(),step)
    xn = np.interp(t, u, x)

    if visualize == True:
        yn = np.interp(t, u, y)

        plt.figure()
        plt.plot(points[0,:],points[1,:], 'r*')
        plt.plot(x,y,'r-')
        plt.plot(xn, yn, 'g*')
        plt.legend(['original points', 'fitted spline', 'calculated equally spaced points'])
        plt.show()

    return xn

def getSurfUniformSpacing(surf_array, zPoints=200, xPoints=200):
    """Defines a uniform (in relation to surface curve) x,z axis spacing 
    based on a 1st order spline fitted to the provided points.\n
    Params:\n
    surf_array - 3d array with a surface (3,xNum,zNum)\n
    step - interpolation step\n
    """
    # https://stackoverflow.com/questions/19117660/how-to-generate-equispaced-interpolating-values
    

    # X Axis
    xMin = np.max(np.min(surf_array[0,:,:],axis=0))
    xMax = np.min(np.max(surf_array[0,:,:],axis=0))
    surf_array_X = np.zeros((3,xPoints,surf_array.shape[2]))
    for i in range(surf_array.shape[2]):
        points = surf_array[:,:,i]
        #Remove repetitions
        points = np.unique(points,axis=1)
        # Sort the points by X axis
        points = points[:,np.argsort(points[0,:])]
        # Calculate spline for the final step and equally starting example values
        ySpl = UnivariateSpline(points[0,:], points[1,:], k=1, s=0)
        zSpl = UnivariateSpline(points[0,:], points[2,:], k=1, s=0)
        # ySpl = scipy.interpolate.interp1d(points[0,:], points[1,:])
        # zSpl = scipy.interpolate.interp1d(points[0,:], points[2,:])
        x = np.linspace(xMin, xMax, xPoints)
        y = ySpl(x)
        # Calculate all all distances between the points 
        # and generate the coordinates on the curve by cumulative summing.
        xd = np.diff(x)
        yd = np.diff(y)
        dist = np.sqrt(xd**2+yd**2)
        u = np.cumsum(dist)
        u = np.hstack([[0],u])
        # Interpolate the x-coordinates independently with respect to the new coordinates.
        t = np.linspace(0,u.max(),xPoints)
        xn = np.interp(t, u, x)
        yn = np.interp(t, u, y)
        zn = zSpl(xn)
        surf_array_X[0,:,i] = xn
        surf_array_X[1,:,i] = yn
        surf_array_X[2,:,i] = zn

    # Z Axis
    zMin = np.max(np.min(surf_array_X[2,:,:],axis=1))
    zMax = np.min(np.max(surf_array_X[2,:,:],axis=1))
    surf_array_Z = np.zeros((3,surf_array_X.shape[1],zPoints))
    for i in range(surf_array_X.shape[1]):
        points = surf_array_X[:,i,:]
        #Remove repetitions
        points = np.unique(points,axis=1)
        # Sort the points by Z axis
        points = points[:,np.argsort(points[2,:])]
        # Calculate spline for the final step and equally starting example values
        # ySpl = UnivariateSpline(points[2,:], points[1,:], k=1, s=0)
        # xSpl = UnivariateSpline(points[2,:], points[0,:], k=1, s=0)
        ySpl = scipy.interpolate.interp1d(points[2,:], points[1,:])
        xSpl = scipy.interpolate.interp1d(points[2,:], points[0,:])
        z = np.linspace(zMin, zMax, zPoints)
        y = ySpl(z)
        # Calculate all all distances between the points 
        # and generate the coordinates on the curve by cumulative summing.
        zd = np.diff(z)
        yd = np.diff(y)
        dist = np.sqrt(zd**2+yd**2)
        u = np.cumsum(dist)
        u = np.hstack([[0],u])
        # Interpolate the x-coordinates independently with respect to the new coordinates.
        t = np.linspace(0,u.max(),zPoints)
        zn = np.interp(t, u, z)
        xn = xSpl(zn)
        yn = np.interp(t, u, y)
        surf_array_Z[0,i,:] = xn
        surf_array_Z[1,i,:] = yn
        surf_array_Z[2,i,:] = zn

    return surf_array_Z


def getMultSurfUniformSpacing(mult_surf_array, zStep=3, xStep=3, mult_normals=None):
    """Defines a uniform (in relation to surface curve) x,z axis spacing 
    based on a 1st order spline fitted to the provided points.\n
    Params:\n
    surf_array - 4d array with surfaces (surfNum,3,xNum,zNum)\n
    step - interpolation step\n
    mult_normals - optional normal vectors matrix to include in calculation
    """
    # https://stackoverflow.com/questions/19117660/how-to-generate-equispaced-interpolating-values
    

    # Sort by x axis
    for i in range(mult_surf_array.shape[0]):
        surf_array = mult_surf_array[i,:,:,:]
        sort_idx = np.argsort(surf_array[0,:,:],axis=0)
        sort_idx_mat = np.array([sort_idx,sort_idx,sort_idx])
        surf_array = np.take_along_axis(surf_array, sort_idx_mat, axis=1)
        mult_surf_array[i,:,:,:] = surf_array
    ### X Axis
    # edge1Val = np.max(np.min(mult_surf_array[:,0,:,:],axis=1))
    # edge2Val = np.min(np.max(mult_surf_array[:,0,:,:],axis=1))

    # Cumulative distance calculation
    xd = np.diff(mult_surf_array[:,0,:,:],axis=1)
    yd = np.diff(mult_surf_array[:,1,:,:],axis=1)
    distMat = np.sqrt(xd**2+yd**2)
    u = np.cumsum(distMat, axis=1)
    # Find average distance
    avgDist = np.mean(np.sum(distMat, axis=1))
    # Define corrrect number of t points with regards to min dist
    tPoints = int(np.round(avgDist/xStep))

    mult_surf_array_X = np.zeros((mult_surf_array.shape[0],3,tPoints,mult_surf_array.shape[3]))
    if mult_normals is not None:
        mult_normals_X = np.zeros((mult_normals.shape[0],3,tPoints,mult_normals.shape[3]))
    for j in range(mult_surf_array.shape[0]):
        surf_array = mult_surf_array[j,:,:,:]
        if mult_normals is not None:
            normals = mult_normals[j,:,:,:]
        edge1Val = np.max(np.min(surf_array[0,:,:],axis=0))    
        edge2Val = np.min(np.max(surf_array[0,:,:],axis=0))
        for i in range(surf_array.shape[2]):
            points = surf_array[:,:,i]
            #Remove repetitions
            points = np.unique(points,axis=1)
            # Sort the points by X axis
            points = points[:,np.argsort(points[0,:])]
            # Calculate spline for the final step and equally starting example values
            zSpl = UnivariateSpline(points[0,:], points[2,:], k=1, s=0)
            ySpl = UnivariateSpline(points[0,:], points[1,:], k=1, s=0)
            x = np.linspace(edge1Val, edge2Val, tPoints)
            y = ySpl(x)
            # Calculate all all distances between the points 
            # and generate the coordinates on the curve by cumulative summing.
            xd = np.diff(x)
            yd = np.diff(y)
            dist = np.sqrt(xd**2+yd**2)
            u = np.cumsum(dist)
            u = np.hstack([[0],u])
            # Interpolate the x-coordinates independently with respect to the new coordinates.
            t = np.linspace(0,u.max(),tPoints)
            xn = np.interp(t, u, x)
            yn = np.interp(t, u, y)
            zn = zSpl(xn)
            mult_surf_array_X[j,0,:,i] = xn
            mult_surf_array_X[j,1,:,i] = yn
            mult_surf_array_X[j,2,:,i] = zn
            if mult_normals is not None:
                norm_points = normals[:,:,i]
                nxSpline = UnivariateSpline(points[0,:], norm_points[0,:], k=1, s=0)
                nySpline = UnivariateSpline(points[0,:], norm_points[1,:], k=1, s=0)
                nzSpline = UnivariateSpline(points[0,:], norm_points[2,:], k=1, s=0)
                mult_normals_X[j,0,:,i] = nxSpline(xn)
                mult_normals_X[j,1,:,i] = nySpline(xn)
                mult_normals_X[j,2,:,i] = nzSpline(xn)

    ### Z Axis
    # edge1Val = np.max(np.min(mult_surf_array_X[:,2,:,:],axis=2))
    # edge2Val = np.min(np.max(mult_surf_array_X[:,2,:,:],axis=2))
    # Cumulative distance calculation
    zd = np.diff(mult_surf_array_X[:,2,:,:],axis=2)
    yd = np.diff(mult_surf_array_X[:,1,:,:],axis=2)
    distMat = np.sqrt(zd**2+yd**2)
    u = np.cumsum(distMat, axis=2)
    # Find average distance
    avgDist = np.mean(np.sum(distMat, axis=2))
    # Define corrrect number of t points with regards to min dist
    tPoints = int(np.round(avgDist/zStep))

    mult_surf_array_Z = np.zeros((mult_surf_array_X.shape[0],3,mult_surf_array_X.shape[2],tPoints))
    if mult_normals is not None:
        mult_normals_Z = np.zeros((mult_normals_X.shape[0],3,mult_normals_X.shape[2],tPoints))
    for j in range(mult_surf_array_X.shape[0]):
        surf_array = mult_surf_array_X[j,:,:,:]
        if mult_normals is not None:
            normals = mult_normals_X[j,:,:,:]
        edge1Val = np.max(np.min(surf_array[2,:,:],axis=1))
        edge2Val = np.min(np.max(surf_array[2,:,:],axis=1))
        for i in range(surf_array.shape[1]):
            points = surf_array[:,i,:]
            #Remove repetitions
            points = np.unique(points,axis=1)
            # Sort the points by Z axis
            points = points[:,np.argsort(points[2,:])]
            # Calculate spline for the final step and equally starting example values
            ySpl = UnivariateSpline(points[2,:], points[1,:], k=1, s=0)
            xSpl = UnivariateSpline(points[2,:], points[0,:], k=1, s=0)
            # ySpl = scipy.interpolate.interp1d(points[2,:], points[1,:])
            # xSpl = scipy.interpolate.interp1d(points[2,:], points[0,:])
            z = np.linspace(edge1Val, edge2Val, tPoints)
            y = ySpl(z)
            # Calculate all all distances between the points 
            # and generate the coordinates on the curve by cumulative summing.
            zd = np.diff(z)
            yd = np.diff(y)
            dist = np.sqrt(zd**2+yd**2)
            u = np.cumsum(dist)
            u = np.hstack([[0],u])
            # Interpolate the x-coordinates independently with respect to the new coordinates.
            t = np.linspace(0,u.max(),tPoints)
            zn = np.interp(t, u, z)
            xn = xSpl(zn)
            yn = np.interp(t, u, y)
            mult_surf_array_Z[j,0,i,:] = xn
            mult_surf_array_Z[j,1,i,:] = yn
            mult_surf_array_Z[j,2,i,:] = zn
            if mult_normals is not None:
                norm_points = normals[:,i,:]
                nxSpline = UnivariateSpline(points[2,:], norm_points[0,:], k=1, s=0)
                nySpline = UnivariateSpline(points[2,:], norm_points[1,:], k=1, s=0)
                nzSpline = UnivariateSpline(points[2,:], norm_points[2,:], k=1, s=0)
                mult_normals_Z[j,0,i,:] = nxSpline(zn)
                mult_normals_Z[j,1,i,:] = nySpline(zn)
                mult_normals_Z[j,2,i,:] = nzSpline(zn)

    if mult_normals is None:
        return mult_surf_array_Z
    else:
        return [mult_surf_array_Z, mult_normals_Z]

def getLinesUniformInterpSpacing(lines_list, zStep=1, xStep=1):
    """Defines a uniform (in relation to surface curve) x,z axis spacing 
    based on a 1st order spline fitted to the provided list of points.\n
    Params:\n
    surf_array - 3d array with a surface (3,xNum,zNum)\n
    step - interpolation step\n
    """
    # https://stackoverflow.com/questions/19117660/how-to-generate-equispaced-interpolating-values
    

    # X Axis
    #Find min max
    minArr = []
    maxArr = []
    for i in range(len(lines_list)):
        minArr.append(np.min(lines_list[i][0,:]))
        maxArr.append(np.max(lines_list[i][0,:]))
    xMin = np.max(np.array(minArr))
    xMax = np.min(np.array(maxArr))
    # First loop to estimate number of interpolation points
    lengthList = []
    for i in range(len(lines_list)):
        points = lines_list[i]
        #Remove repetitions
        points = np.unique(points,axis=1)
        # Sort the points by X axis
        points = points[:,np.argsort(points[0,:])]
        # Calculate spline for the final step and equally starting example values
        ySpl = UnivariateSpline(points[0,:], points[1,:], k=1, s=0)
        x = np.arange(xMin, xMax, xStep)
        y = ySpl(x)
        xd = np.diff(x)
        yd = np.diff(y)
        dist = np.sqrt(xd**2+yd**2)
        u = np.cumsum(dist)
        u = np.hstack([[0],u])
        # Interpolate the x-coordinates independently with respect to the new coordinates.
        t = np.arange(0,u.max(),xStep)
        lengthList.append(t.shape[0])
    
    xPoints = int(np.round(np.mean(np.array(lengthList))))
    # Second "proper" loop
    surf_array_X = np.zeros((3,xPoints,len(lines_list)))
    for i in range(len(lines_list)):
        points = lines_list[i]
        #Remove repetitions
        points = np.unique(points,axis=1)
        # Sort the points by X axis
        points = points[:,np.argsort(points[0,:])]
        # Calculate spline for the final step and equally starting example values
        ySpl = UnivariateSpline(points[0,:], points[1,:], k=1, s=0)
        zSpl = UnivariateSpline(points[0,:], points[2,:], k=1, s=0)
        # ySpl = scipy.interpolate.interp1d(points[0,:], points[1,:])
        # zSpl = scipy.interpolate.interp1d(points[0,:], points[2,:])
        x = np.arange(xMin, xMax, xStep)
        y = ySpl(x)
        # Calculate all all distances between the points 
        # and generate the coordinates on the curve by cumulative summing.
        xd = np.diff(x)
        yd = np.diff(y)
        dist = np.sqrt(xd**2+yd**2)
        u = np.cumsum(dist)
        u = np.hstack([[0],u])
        # Interpolate the x-coordinates independently with respect to the new coordinates.
        t = np.linspace(0,u.max(),xPoints)
        xn = np.interp(t, u, x)
        yn = np.interp(t, u, y)
        zn = zSpl(xn)
        surf_array_X[0,:,i] = xn
        surf_array_X[1,:,i] = yn
        surf_array_X[2,:,i] = zn

    # Z Axis
    zMin = np.max(np.min(surf_array_X[2,:,:],axis=1))
    zMax = np.min(np.max(surf_array_X[2,:,:],axis=1))
    # TODO: fix so that +1 is not needed
    zPoints = np.arange(zMin, zMax+1, zStep).shape[0]
    surf_array_Z = np.zeros((3,surf_array_X.shape[1],zPoints))
    for i in range(surf_array_X.shape[1]):
        points = surf_array_X[:,i,:]
        #Remove repetitions
        points = np.unique(points,axis=1)
        # Sort the points by Z axis
        points = points[:,np.argsort(points[2,:])]
        # Calculate spline for the final step and equally starting example values
        ySpl = UnivariateSpline(points[2,:], points[1,:], k=1, s=0)
        xSpl = UnivariateSpline(points[2,:], points[0,:], k=1, s=0)
        # ySpl = scipy.interpolate.interp1d(points[2,:], points[1,:])
        # xSpl = scipy.interpolate.interp1d(points[2,:], points[0,:])
        z = np.arange(zMin, zMax+1, zStep)
        y = ySpl(z)
        x = xSpl(z)
        surf_array_Z[0,i,:] = x
        surf_array_Z[1,i,:] = y
        surf_array_Z[2,i,:] = z

    return surf_array_Z

def rescaleImage(img, minVal, maxVal):
    """Rescales input image to a given range.\n
    Params:\n
    img - image to rescale\n
    minVal - minimum pixel value\n
    maxVal - maximum pixel value.
    """
    img = ((img - img.min()) * (minVal/(img.max() - img.min()) * maxVal)).astype(np.int32)
    return img


def layersToMatrix(layersList, imgDim):
    """Converts list of layers to a 3d matrix with all layer points marked.\n
    Params:\n
    layersList - list of layers from honeycomb surface detection\n
    minVal - dimensions of the 3D image stack.
    """
    segmImg = np.zeros(imgDim).astype('uint8')

    for i in range(len(layersList)):
        zStacks = layersList[i]
        for j in range(len(zStacks)):
            foldedSurfaces = zStacks[j]
            for k in range(len(foldedSurfaces)):
                foldedSurface = foldedSurfaces[k]
                for m in range(foldedSurface.shape[1]):
                    segmImg[j,int(foldedSurface[0,m]),int(foldedSurface[1,m])] = 1

    return segmImg

def layersToSurface(layersList):
    """Converts list of layers to a list of surfaces as a preparation for 
    creating a mesh for visualization.\n
    Params:\n
    layersList - list of layers from honeycomb surface detection
    """
    surfaceList = []
    for i in range(len(layersList)):
        zStacks = layersList[i]
        X = np.zeros((2,zStacks[0][0].shape[1],len(zStacks)))
        Y = np.zeros((2,zStacks[0][0].shape[1],len(zStacks)))
        Z = np.zeros((2,zStacks[0][0].shape[1],len(zStacks)))
        for j in range(len(zStacks)):
            foldedSurfaces = zStacks[j]
            # Helper surfaces are not taken into account
            for k in range(2):
                foldedSurface = foldedSurfaces[k]
                for m in range(foldedSurface.shape[1]):
                    X[k,m,j] = foldedSurface[0,m]
                    Y[k,m,j] = foldedSurface[1,m]
                    Z[k,m,j] = j
        XYZList = [X[0,:,:], Y[0,:,:], Z[0,:,:]]
        surfaceList.append(XYZList)
        XYZList = [X[1,:,:], Y[1,:,:], Z[1,:,:]]
        surfaceList.append(XYZList)

    return surfaceList

    
def saveHcPoints(hcList, savePath):

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

def loadHcPoints(hcList, loadPath):

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
