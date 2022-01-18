# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:20:37 2022

@author: pawel
"""


import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as scp
import scipy
import cv2 as cv

class HoneycombUnfold2d:
    lines = np.array([])
    lines_interp = np.empty((2,0))
    normals = np.empty((2,2,0))
    unfolding_points = np.empty((2,0))

    interp_points=0

    def __init__(self, img, vis_img, visualize=True):
        self.img = img
        self.vis_img = vis_img
        self.img_shape = img.shape
        self.visualize = visualize

        if visualize == True:
            self.fig, self.ax = plt.subplots(2,2)

    def draw_corners(self):

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
            self.ax[0,0].imshow(self.img, cmap='gray')
            self.ax[0,0].plot(self.lines[0,:],self.lines[1,:], '-')

        for i in range(self.lines.shape[1]-1):
            start_point = (int(np.round(self.lines[0,i])), int(np.round(self.lines[1,i])))
            end_point = (int(np.round(self.lines[0,i+1])), int(np.round(self.lines[1,i+1])))
            color = (0, 0, 255)
            thickness = 9
            self.vis_img = cv.line(self.vis_img, start_point, end_point, color, thickness)

        return self.vis_img

    def interpolate_points(self,points_scale=1):
        if self.lines.size == 0 or self.lines.shape[1] < 2:
            raise Exception('No lines are defined. Define the line first with draw_line')
        for i in range(self.lines.shape[1]-1):
            # Get linterpolation start and end points
            x1 = self.lines[0,i]
            x2 = self.lines[0,i+1]
            y1 = self.lines[1,i]
            y2 = self.lines[1,i+1]
            dist = np.linalg.norm(np.array((x1,y1))-np.array((x2,y2)))
            # Interpolate x and y
            x_interp = np.linspace(x1,x2,int(dist*points_scale))
            y_interp = np.linspace(y1,y2,int(dist*points_scale))
            # Merge and append to main vector
            line_interp = np.array([x_interp,y_interp])
            self.lines_interp = np.hstack((self.lines_interp,line_interp))

        if self.visualize == True:
            self.ax[0,1].imshow(self.img, cmap='gray')
            self.ax[0,1].plot(self.lines_interp[0,:],self.lines_interp[1,:], '*',markersize=1)

    def smooth_interp_corners(self):
        if self.lines_interp.size == 0:
            raise Exception('Interpolated points are not defined')
        # Pass x and y through gaussian smoothing filter
        x_smooth = scp.gaussian_filter1d(self.lines_interp[0,:],3)
        y_smooth = scp.gaussian_filter1d(self.lines_interp[1,:],3)
        self.lines_interp = np.array([x_smooth,y_smooth])

        if self.visualize == True:
            self.ax[1,0].imshow(self.img, cmap='gray')
            self.ax[1,0].plot(self.lines_interp[0,:],self.lines_interp[1,:], '*',markersize=1)
            # plt.show()

    def calculate_normals(self,normals_range=30):
        if self.lines_interp.size == 0:
            raise Exception('Interpolated points are not defined')
        for i in range(self.lines_interp.shape[1]-1):
            # Calculate direction vector for the point
            if i == 0:
                vec = self.lines_interp[:,i+1] - self.lines_interp[:,i]
            elif i == self.lines_interp.shape[1]-1:
                vec = self.lines_interp[:,i] - self.lines_interp[:,i-1]
            else:
                vec = self.lines_interp[:,i+1] - self.lines_interp[:,i-1]
            # Normalize vector
            vec = vec/np.linalg.norm(vec)
            # Calculate perpendicular vector
            per_vec = np.empty_like(vec)
            per_vec[0] = -vec[1]
            per_vec[1] = vec[0]
            # Create 2 points moved by the vector in either direction
            xy1 = self.lines_interp[:,i] + normals_range*per_vec
            xy2 = self.lines_interp[:,i] - normals_range*per_vec
            # create normal and add to vector
            normal = np.array((xy1,xy2))
            self.normals = np.dstack((self.normals,normal))
        
        if self.visualize == True:
            self.ax[1,1].imshow(self.img, cmap='gray')
            for i in range(self.normals.shape[2]):
                self.ax[1,1].plot(self.normals[:,0,i],self.normals[:,1,i], '-',color='k')
            plt.show()

    def get_unfold_points_from_normals(self, interp_points=60):
        if self.normals.shape[2] == 0:
            raise Exception('normals are not defined')

        self.interp_points = interp_points
        for i in range(self.normals.shape[2]):
            # Get linterpolation start and end points
            x1 = self.normals[1,0,i]
            x2 = self.normals[0,0,i]
            y1 = self.normals[1,1,i]
            y2 = self.normals[0,1,i]
            # Interpolate x and y
            x_interp = np.linspace(x1,x2,interp_points)
            y_interp = np.linspace(y1,y2,interp_points)
            # Merge and append to main vector
            line_interp = np.array([x_interp,y_interp])
            self.unfolding_points = np.hstack((line_interp,self.unfolding_points))

    def unfold_image(self):
        if self.unfolding_points.shape[1] == 0:
            raise Exception('Unfolding points are not defined')

        x = np.linspace(0,self.img_shape[0]-1,self.img_shape[0])
        y = np.linspace(0,self.img_shape[1]-1,self.img_shape[1])
        img_interp = scipy.interpolate.RegularGridInterpolator((y, x), self.img.transpose(), method='linear')
        unfolded_img = img_interp(self.unfolding_points.transpose())
        unfolded_img = unfolded_img.reshape((self.normals.shape[2],self.interp_points)).transpose()

        return unfolded_img.astype(np.int32)
        
    def fold_surfaces_back(self, surfaces, interpolate=True):
        """Folds surfaces back to original shape. Returns a line with original data points 
        or with one (rounded) value for each x axis pixel, if interpolate=True"""
        if self.unfolding_points.shape[1] == 0:
            raise Exception('Unfolding points are not defined')
        # Create "coordinate image" out of normals unfolding points
        unfolding_points_mat = self.unfolding_points.reshape(2,self.normals.shape[2],self.interp_points)
        folded_surfaces = []
        for i in range(len(surfaces)):
            folded_surface = np.empty((2,0))
            surface = surfaces[i]
            for j in range(surface.shape[0]):
                # Get point position on original image
                point = unfolding_points_mat[:,j,surface[j]]
                # Append to surface vec
                folded_surface = np.hstack((folded_surface,np.expand_dims(point,axis=1)))
            # Sort surface ascending in relation to x axis
            folded_surface = folded_surface[:,np.argsort(folded_surface[0, :])]
            if interpolate == True:
                # Create interpolated surface
                f = scipy.interpolate.interp1d(folded_surface[0, :], folded_surface[1, :])
                xnew = np.arange(np.ceil(folded_surface[0, 0]),np.floor(folded_surface[0, -1]),1)
                ynew = np.round(f(xnew))
                folded_surface = np.array([xnew,ynew])

            folded_surfaces.append(folded_surface)
        return folded_surfaces


class HoneycombUnfold3d:
    lines = np.empty((3,0,0))
    lines_interp = np.empty((2,0))
    normals = np.empty((2,2,0))
    unfolding_points = np.empty((2,0))

    interp_points=0

    def __init__(self, img, vis_img, visualize=True):
        self.img = img
        self.vis_img = vis_img
        self.img_shape = img.shape
        self.visualize = visualize
        self.sliceIdx = [0, int(self.img_shape[0]/2), self.img_shape[0]]


    def draw_corners(self):

        sliceList = ["Bottom", "Middle", "Top"]

        for i in range(3):

            plt.figure()
            plt.imshow(self.vis_img[self.sliceIdx[i],:,:], cmap='gray')
            plt.suptitle(f"{sliceList[i]} slice: Click on the corners of folds of the chosen unmarked layer from left to right.")
            plt.title("Left button - new point, Right button - remove point, Middle button - end process", fontsize=8)

            ax = plt.gca()
            xy = plt.ginput(-1, timeout=60)
            plt.close()

            x = [p[0] for p in xy]
            y = [p[1] for p in xy]
            z = [self.sliceIdx[i] for j in range(len(x))]
            self.lines = np.dstack((self.lines,np.array([x,y,z])))

            for j in range(self.lines.shape[1]-1):
                start_point = (int(np.round(self.lines[i,0,j])), int(np.round(self.lines[i,1,j])))
                end_point = (int(np.round(self.lines[i,0,j+1])), int(np.round(self.lines[i,1,j+1])))
                color = (0, 0, 255)
                thickness = 9
                self.vis_img[self.sliceIdx[i],:,:] = cv.line(self.vis_img[self.sliceIdx[i],:,:], start_point, end_point, color, thickness)

        if self.visualize == True:
            for i in range(3):
                plt.figure()
                plt.subplot(1,3,i)
                plt.imshow(self.img[self.sliceIdx[i],:,:], cmap='gray')
                plt.plot(self.lines[i,0,:],self.lines[i,1,:], '-')
                
        return self.vis_img