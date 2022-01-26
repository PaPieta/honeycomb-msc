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

import helpers


class HoneycombUnfoldSlicewise3d:
    lines = []
    lines_interp = np.empty((2,0))
    normals = np.empty((2,3,0))
    unfolding_points = np.empty((3,0))

    interp_points = 0
    interp_x_len = 0
    interp_z_len = 0

    def __init__(self, img, vis_img, visualize=True):
        self.img = img
        self.vis_img = vis_img
        self.img_shape = img.shape
        self.visualize = visualize
        self.sliceIdx = [0, int(self.img_shape[0]/2), self.img_shape[0]-1]


    def draw_corners(self):

        sliceList = ["Bottom", "Middle", "Top"]
        # Loop through the slices (top, middle, bottom)
        for i in range(3):
            # Create image for pointing the corners
            plt.figure()
            plt.imshow(self.vis_img[self.sliceIdx[i],:,:], cmap='gray')
            plt.suptitle(f"{sliceList[i]} slice: Click on the corners of folds of the chosen unmarked layer from left to right.")
            plt.title("Left button - new point, Right button - remove point, Middle button - end process", fontsize=8)

            ax = plt.gca()
            xy = plt.ginput(-1, timeout=60)
            plt.close()
            # Get point coordinates and append to a list
            x = [p[0] for p in xy]
            y = [p[1] for p in xy]
            z = [self.sliceIdx[i] for j in range(len(x))]
            line = np.array([x,y,z])
            self.lines.append(line)
            # Draw the lines onto the visualization image (needed for multiple surface caluclation)
            for j in range(line.shape[1]-1):
                start_point = (int(np.round(line[0,j])), int(np.round(line[1,j])))
                end_point = (int(np.round(line[0,j+1])), int(np.round(line[1,j+1])))
                color = (0, 0, 255)
                thickness = 9
                self.vis_img[self.sliceIdx[i],:,:] = cv.line(self.vis_img[self.sliceIdx[i],:,:], start_point, end_point, color, thickness)
        # Plot the results of the corner drawing
        if self.visualize == True:
            plt.figure()
            for i in range(3):
                line = self.lines[i]
                plt.subplot(1,3,i+1)
                plt.imshow(self.img[self.sliceIdx[i],:,:], cmap='gray')
                plt.plot(line[0,:],line[1,:], '-')

            plt.show()
                
        return self.vis_img


    def interpolate_points(self,step=1):
        if len(self.lines) == 0:
            raise Exception('No lines are defined. Define the line first with draw_line')

        x_low_list = []
        x_high_list = []
        linesArray = np.empty((3,0))
        for i in range(len(self.lines)):
            linesArray = np.hstack((linesArray,self.lines[i]))
            x_low_list.append(np.min(self.lines[i][0,:]))
            x_high_list.append(np.max(self.lines[i][0,:]))

        x_low = np.max(np.array(x_low_list))
        x_high = np.min(np.array(x_high_list))
        
        # we want to get y values based on x and z
        # Define interpolation points and get y values
        zUnique = np.arange(0,self.img_shape[0],step)

        xnew = np.empty(0)
        znew = np.empty(0)
        for i in range(zUnique.shape[0]):
            xtemp = np.arange(x_low,x_high,step)
            ztemp = np.full(xtemp.shape[0],zUnique[i])

            xnew = np.hstack((xnew,xtemp))
            znew = np.hstack((znew,ztemp))
        # save for calculating normals
        self.interp_x_len = xtemp.shape[0]
        self.interp_z_len = zUnique.shape[0]

        xi = np.array((xnew,znew)).transpose()
        points = np.array((linesArray[0, :],linesArray[2, :])).transpose()
        # create interpolation function  x,z,y order
        ynew = scipy.interpolate.griddata(points,linesArray[1, :], xi)
        self.lines_interp = np.array((xnew,ynew,znew))


        if self.visualize == True:
            plt.figure()
            for i in range(3):
                # Find interpolated z layer closest to the original image layers
                zlayer = np.argmin(np.abs(zUnique-self.sliceIdx[i]))
                # Get points for the layer
                interp_points = self.lines_interp[0:2,zlayer*self.interp_x_len:(zlayer+1)*self.interp_x_len]

                plt.subplot(1,3,i+1)
                plt.imshow(self.img[self.sliceIdx[i],:,:], cmap='gray')
                plt.plot(interp_points[0,:],interp_points[1,:], '*',markersize=4)
            plt.show()

    
    def smooth_interp_corners(self):
        if self.lines_interp.size == 0:
            raise Exception('Interpolated points are not defined')
        # Pass x and y through gaussian smoothing filter
        x_smooth = np.empty(0)
        y_smooth = np.empty(0)
        for i in range(self.interp_z_len):
            start = i*self.interp_x_len
            end = (i+1)*self.interp_x_len
            x_temp = scp.gaussian_filter1d(self.lines_interp[0,start:end],2)
            y_temp = scp.gaussian_filter1d(self.lines_interp[1,start:end],2)
            x_smooth = np.hstack((x_smooth,x_temp))
            y_smooth = np.hstack((y_smooth,y_temp))
        self.lines_interp = np.array([x_smooth,y_smooth,self.lines_interp[2,:]])

        if self.visualize == True:
            zUnique = np.unique(self.lines_interp[2,:])
            plt.figure()
            for i in range(3):
                # Find interpolated z layer closest to the original image layers
                zlayer = np.argmin(np.abs(zUnique-self.sliceIdx[i]))
                # Get points for the layer
                interp_points = self.lines_interp[0:2,zlayer*self.interp_x_len:(zlayer+1)*self.interp_x_len]

                plt.subplot(1,3,i+1)
                plt.imshow(self.img[self.sliceIdx[i],:,:], cmap='gray')
                plt.plot(interp_points[0,:],interp_points[1,:], '*',markersize=1)
            plt.show()

    
    def calculate_normals(self,normals_range=30):
        if self.lines_interp.size == 0:
            raise Exception('Interpolated points are not defined')
        xCount=0
        zCount=0
        for i in range(self.lines_interp.shape[1]-1):
            # Get triangle points x coords
            if xCount == 0:
                pt1_x = 0
                pt2_x = 0
                pt3_x = 1
            elif xCount == self.interp_x_len-1:
                pt1_x = 0
                pt2_x = -1
                pt3_x = 0
            else:
                pt1_x = 0
                pt2_x = -1
                pt3_x = 1
            # Get triangle points z coords
            if zCount == 0:
                pt1_z = 0
                pt2_z = self.interp_x_len
                pt3_z = self.interp_x_len
            elif zCount == self.interp_z_len-1:
                pt1_z = -self.interp_x_len
                pt2_z = 0
                pt3_z = 0
            else:
                pt1_z = -self.interp_x_len
                pt2_z = self.interp_x_len
                pt3_z = self.interp_x_len
            # Combine chosen coordinates to create points
            pt1 = self.lines_interp[:,i+pt1_x+pt1_z]
            pt2 = self.lines_interp[:,i+pt2_x+pt2_z]
            pt3 = self.lines_interp[:,i+pt3_x+pt3_z]
            # Create a list and calculate normal vector
            poly = [pt1, pt2, pt3]
            normVec = np.array(helpers.surfaceNormal(poly))
            # Normalize vector
            normVec = normVec/np.linalg.norm(normVec)
            # Create 2 points moved by the vector in either direction
            xyz1 = self.lines_interp[:,i] + normals_range*normVec
            xyz2 = self.lines_interp[:,i] - normals_range*normVec
            # create normal line and add to vector
            normal = np.array((xyz1,xyz2))
            self.normals = np.dstack((self.normals,normal))
            # Update counters
            xCount+=1
            if xCount == self.interp_x_len:
                xCount = 0
                zCount +=1
        
        if self.visualize == True:
            zUnique = np.unique(self.lines_interp[2,:])
            plt.figure()
            for i in range(3):
                # Find interpolated z layer closest to the original image layers
                zlayer = np.argmin(np.abs(zUnique-self.sliceIdx[i]))
                # Get normals for the layer
                temp_normals = self.normals[:,0:2,zlayer*self.interp_x_len:(zlayer+1)*self.interp_x_len]

                plt.subplot(1,3,i+1)
                plt.imshow(self.img[self.sliceIdx[i],:,:], cmap='gray')
                for j in range(temp_normals.shape[2]):
                    plt.plot(temp_normals[:,0,j],temp_normals[:,1,j], '-',color='k')
            plt.show()