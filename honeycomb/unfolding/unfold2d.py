# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:20:37 2022

@author: Pawel Pieta s202606@student.dtu.dk
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as scp
import scipy
import cv2 as cv
from scipy.interpolate import UnivariateSpline

class Unfold2d:
    """Set of methods for unfolding and folding a slice of the honeycomb scan
    for purposes of layered surfaces detection on a 2D image.\n
    Pipeline of the segmentation process:\n
    draw_corners -> interpolate_points -> smooth_interp-corners ->
    calculate_normals -> get_unfold_points_from_normals -> unfold_image ->
    (external) segment layered surfaces -> fold_surfaces_back."""

    def __init__(self, img, vis_img, visualize=True):
        """Class initialization. 
        Params:\n
        img - 3D image stack of the honeycmb scan\n
        vis_img - deep copy of the image stack for visualization purposes\n
        visualize - if True, will visualize each step of the unfolding process\n
        Attributes:\n
        img, vis_img, visualize - see above\n
        img_shape - shape of the input image\n
        lines - list of corner points marked on the image representing the lines along the honeycomb structure\n
        lines_interp - array of points interpolated along the lines
        normals - array of lines normal to the orignal structure line, calculated for each interpolated point\n
        unfolding_points - points interpolated along the normal lines, used for unfolding of the image\n
        interp_points - number of interpolation points along each normal line\n
        """
        self.img = img
        self.vis_img = vis_img
        self.visualize = visualize
        self.img_shape = img.shape

        self.lines = np.array([])
        self.lines_interp = np.empty((2,0))
        self.normals = np.empty((2,2,0))
        self.unfolding_points = np.empty((2,0))

        self.interp_points=0

        self.unfolded_img = np.empty((1,0))



    def draw_corners(self):
        """Opens interactive plot on a provided image to allow user to draw
        corners of a chosen honeycomb surface.
        """
        plt.figure()
        plt.imshow(self.vis_img, cmap='gray')
        plt.suptitle("Click on the corners of one of the unmarked folds from left to right.")
        plt.title("Left button - new point, Right button - remove point, Middle button - end process", fontsize=8)

        ax = plt.gca()
        xy = plt.ginput(-1, timeout=60)
        plt.close()

        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        self.lines = np.array([x,y])

        if self.visualize == True:
            plt.figure()
            plt.imshow(self.img, cmap='gray')
            plt.plot(self.lines[0,:],self.lines[1,:], 'r-')
            plt.plot(self.lines[0,:],self.lines[1,:], 'ro')
            plt.show()

        for i in range(self.lines.shape[1]-1):
            start_point = (int(np.round(self.lines[0,i])), int(np.round(self.lines[1,i])))
            end_point = (int(np.round(self.lines[0,i+1])), int(np.round(self.lines[1,i+1])))
            color = (0, 0, 255)
            thickness = 9
            self.vis_img = cv.line(self.vis_img, start_point, end_point, color, thickness)

        return self.vis_img


    def interpolate_points(self,step=1):
        """interpolates points along marked corners with a chosen scale.\n
        Params:\n
        step - interpolation step distance
        """

        #Remove repetitions
        points = np.unique(self.lines,axis=1)
        # Sort the points by X axis
        points = points[:,np.argsort(points[0,:])]
        # Calculate spline for the final step and equally starting example values
        ySpl = UnivariateSpline(points[0,:], points[1,:], k=1, s=0)
        x = np.arange(np.min(self.lines[0,:]), np.max(self.lines[0,:]), step)
        y = ySpl(x)
        xd = np.diff(x)
        yd = np.diff(y)
        dist = np.sqrt(xd**2+yd**2)
        u = np.cumsum(dist)
        u = np.hstack([[0],u])
        # Interpolate the x-coordinates independently with respect to the new coordinates.
        t = np.arange(0,u.max(),step)
        xn = np.interp(t, u, x)
        yn = np.interp(t, u, y)
        self.lines_interp = np.array([xn,yn])

        if self.visualize == True:
            
            # plt.figure()
            # plt.imshow(self.img[80:550,:], cmap='gray')
            # plt.plot(x[::6],y[::6]-80,'bo',markersize=4, fillstyle='none')
            # plt.plot(xn[::6], yn[::6]-80, 'ro',markersize=2)
            # plt.legend(['Points equally spaced in relation to X Axis', 'Points equally spaced along the honeycomb wall'])
            # plt.show()

            plt.figure()
            plt.imshow(self.img, cmap='gray')
            plt.plot(self.lines_interp[0,::6],self.lines_interp[1,::6], 'ro',markersize=2)
            plt.show()


    def smooth_interp_corners(self):
        """Applies smoothing on the interpolation points along X and Y axis 
        to provide a smoother transition in the corners of the strucuture.
        """
        if self.lines_interp.size == 0:
            raise Exception('Interpolated points are not defined')
        # Pass x and y through gaussian smoothing filter
        x_smooth = scp.gaussian_filter1d(self.lines_interp[0,:],3)
        y_smooth = scp.gaussian_filter1d(self.lines_interp[1,:],3)
        self.lines_interp = np.array([x_smooth,y_smooth])

        if self.visualize == True:
            plt.figure()
            plt.imshow(self.img, cmap='gray')
            # plt.plot(self.lines_interp[0,::2],self.lines_interp[1,::2],'bo',markersize=8, fillstyle='none')
            # plt.plot(x_smooth[::2],y_smooth[::2], 'ro',markersize=4)
            # plt.legend(['Points before smoothing', 'Points after smoothing'])
            plt.plot(self.lines_interp[0,::6],self.lines_interp[1,::6], 'ro',markersize=2)
            plt.show()


    def calculate_normals(self,normals_range=30):
        """Calculate lines normal to the interpolated points with given range.\n
        Params:\n
        normals_range - length range of each normal line (length = 2*range)
        """
        if self.lines_interp.size == 0:
            raise Exception('Interpolated points are not defined')

        vecMat = np.zeros_like(self.lines_interp)
        # Fill central values
        vecMat[:,1:-1] = self.lines_interp[:,2:] - self.lines_interp[:,:-2]
        # Fill first and last values 
        vecMat[:,0] = self.lines_interp[:,1] - self.lines_interp[:,0]
        vecMat[:,-1] = self.lines_interp[:,-1] - self.lines_interp[:,-2]
        # Normalize
        vecMat = vecMat/np.apply_along_axis(np.linalg.norm,0,vecMat)
        # Calculate perpendicular vector
        perVecMat = np.empty_like(vecMat)
        perVecMat[0,:] = -vecMat[1,:]
        perVecMat[1,:] = vecMat[0,:]
        # Calculate points moved by vector in both directions
        normP1 = self.lines_interp - perVecMat*normals_range
        normP2 = self.lines_interp + perVecMat*normals_range
        # Combine points
        self.normals = np.array([normP1,normP2])
        
        if self.visualize == True:
            plt.figure()
            plt.imshow(self.img, cmap='gray')
            for i in range(0,self.normals.shape[2],6):
                plt.plot(self.normals[:,0,i],self.normals[:,1,i], '-',color='k')
            plt.show()


    def get_unfold_points_from_normals(self, interp_points=60):
        """Interpolates points along each normal line to create unfolding points.\n
        Params:\n
        interp_points - number of points interpolated along each normal line.
        """
        if self.normals.shape[2] == 0:
            raise Exception('normals are not defined')

        self.interp_points = interp_points
        for i in range(self.normals.shape[2]):
            # Get linterpolation start and end points
            x1 = self.normals[0,1,i]
            x2 = self.normals[1,1,i]
            y1 = self.normals[0,0,i]
            y2 = self.normals[1,0,i]
            # Interpolate x and y
            # x_interp = np.round(np.linspace(x1,x2,interp_points))
            # y_interp = np.round(np.linspace(y1,y2,interp_points))
            x_interp = np.linspace(x1,x2,interp_points)
            y_interp = np.linspace(y1,y2,interp_points)
            # Merge and append to main vector
            line_interp = np.array([x_interp,y_interp])
            self.unfolding_points = np.hstack((self.unfolding_points,line_interp))

        if self.visualize == True:
            vis_unf_points = np.reshape(self.unfolding_points,(2,self.normals.shape[2],interp_points))
            plt.figure()
            plt.imshow(self.img, cmap='gray')
            plt.plot(vis_unf_points[1,::6,::9].ravel(),vis_unf_points[0,::6,::9].ravel(), 'ro',markersize=1)
            plt.show()


    def unfold_image(self, method='linear'):
        """Creates an unfolded image of the honeycomb structure 
        using previously defined unfolding points.
        Params:\n
        method - interpolation method, 'linear' or 'nearest'
        """
        if self.unfolding_points.shape[1] == 0:
            raise Exception('Unfolding points are not defined')

        x = np.linspace(0,self.img_shape[0]-1,self.img_shape[0])
        y = np.linspace(0,self.img_shape[1]-1,self.img_shape[1])
        img_interp = scipy.interpolate.RegularGridInterpolator((x, y), self.img, method=method)
        temp_unf_points = np.swapaxes(self.unfolding_points,0,1)
        unfolded_img = img_interp(temp_unf_points)
        unfolded_img = unfolded_img.reshape((self.normals.shape[2],self.interp_points)).transpose()

        self.unfolded_img = unfolded_img.astype(np.int32)

        if self.visualize == True:
            plt.figure()
            plt.imshow(unfolded_img,cmap='gray')
            plt.show()

        return self.unfolded_img
        

    def fold_surfaces_back(self, surfaces):
        """Folds surfaces back to original shape. Returns a line with original data points\n
        Params:\n
        surfaces - list of surfaces to unfold\n
        zIdx - zAxis index of the surface
        """
        if self.unfolding_points.shape[1] == 0:
            raise Exception('Unfolding points are not defined')
        # Create "coordinate image" out of normals unfolding points
        # temp_unf_points = np.swapaxes(self.unfolding_points,0,1)
        temp_unf_points = self.unfolding_points[[1,0],:]
        unfolding_points_mat = temp_unf_points.reshape(2,self.normals.shape[2],self.interp_points)
        folded_surfaces = []
        for i in range(len(surfaces)):
            folded_surface = np.empty((2,0))
            surface = surfaces[i]
            for j in range(surface.shape[0]):
                # Get point position on original image
                if surface[j].is_integer():
                    point = unfolding_points_mat[:,j,surface[j]]
                else:
                    surfVal = surface[j]
                    #Edge case
                    if unfolding_points_mat.shape[2] - surfVal < 1:
                        p1 = unfolding_points_mat[:,j,int(np.floor(surfVal))]
                    else:
                        p1 = unfolding_points_mat[:,j,int(np.ceil(surfVal))]
                    p2 = unfolding_points_mat[:,j,int(np.floor(surfVal))]
                    point = p1*(np.ceil(surfVal)-surfVal) + p2*(surfVal-np.floor(surfVal))
                # Append to surface vec
                folded_surface = np.hstack((folded_surface,np.expand_dims(point,axis=1)))
            # Sort surface ascending in relation to x axis
            folded_surfaces.append(folded_surface)
        return folded_surfaces

