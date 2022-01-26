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


def imgGaussianModel(img, step_3d=89):
    
    imgFlat = img.ravel()
    # If img is a 3d stack, take only a part of the pixels
    if img.ndim > 2:
        imgLen = imgFlat.shape[0]
        idxList = np.arange(0,imgLen,step_3d)
        imgFlat = np.take(imgFlat, idxList)

    imgFlat = imgFlat[imgFlat != 0]

    gm = GaussianMixture(n_components=2, random_state=0)
    gm.fit(imgFlat.reshape(-1,1))
    means = gm.means_
    variances = np.array([[np.min(gm.covariances_)],[np.min(gm.covariances_)]])

    return means, variances

def gaussianProbFunction(img, mean, variance):
    img = np.abs(img - mean)
    cost = np.exp(-0.5*((img/np.sqrt(variance))**2.0))
    return cost

def sigmoidProbFunction(img, mean, variance, visualize=False):
    sigma = np.sqrt(variance)
    # scale = 1/(np.mean(sigma)/abs(mean[1]-mean[0]))
    gaussCenterProb = 0.999
    center = np.mean(mean)
    scale = np.log(1/gaussCenterProb - 1)/(mean[0]-center)
    
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
    n = [0.0, 0.0, 0.0]

    for i, v_curr in enumerate(poly):
        v_next = poly[(i+1) % len(poly)]
        n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
        n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])

    normalised = [i/sum(n) for i in n]

    return normalised

def getFuncUniformSpacing(points, startX, endX, step=1, visualize=False):
    # Sort the points
    points = points[:,np.argsort(points[0,:])]
    # Make sure there are no points with same x val
    for i in range(points.shape[1]-1):
        if points[0,i] == points[0,i+1]:
            points[0,i] = points[0,i] - 0.0001*i
    # Sort again to be sure 
    points = points[:,np.argsort(points[0,:])]
    # Calculate spline and get example values
    spl = UnivariateSpline(points[0,:], points[1,:], k=1)
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

def rescaleImage(img, minVal, maxVal):
    """Rescales input image to a given range.\n
    Params:\n
    img - image to rescale\n
    minVal - minimum pixel value\n
    maxVal - maximum pixel value.
    """
    img = ((img - img.min()) * (minVal/(img.max() - img.min()) * maxVal)).astype(np.int32)
    return img


def convertLayersToMatrix(layersList, imgDim):

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
    
