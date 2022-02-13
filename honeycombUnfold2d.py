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

class HoneycombUnfold2d:
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

        if visualize == True:
            self.fig, self.ax = plt.subplots(2,2)


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
        """interpolates points along marked corners with a chosen scale.\n
        Params:\n
        points_scale - interpolation step scale (ratio to a pixel)
        """
        if self.lines.size == 0 or self.lines.shape[1] < 2:
            raise Exception('No lines are defined. Define the line first with draw_line')
        for i in range(self.lines.shape[1]-1):
            # Get interpolation start and end points
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
            self.ax[1,0].imshow(self.img, cmap='gray')
            self.ax[1,0].plot(self.lines_interp[0,:],self.lines_interp[1,:], '*',markersize=1)
            # plt.show()


    def calculate_normals(self,normals_range=30):
        """Calculate lines normal to the interpolated points with given range.\n
        Params:\n
        normals_range - length range of each normal line (length = 2*range)
        """
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
        """Interpolates points along each normal line to create unfolding points.\n
        Params:\n
        interp_points - number of points interpolated along each normal line.
        """
        if self.normals.shape[2] == 0:
            raise Exception('normals are not defined')

        self.interp_points = interp_points
        for i in range(self.normals.shape[2]):
            # Get linterpolation start and end points
            x1 = self.normals[1,1,i]
            x2 = self.normals[0,1,i]
            y1 = self.normals[1,0,i]
            y2 = self.normals[0,0,i]
            # Interpolate x and y
            # x_interp = np.round(np.linspace(x1,x2,interp_points))
            # y_interp = np.round(np.linspace(y1,y2,interp_points))
            x_interp = np.linspace(x1,x2,interp_points)
            y_interp = np.linspace(y1,y2,interp_points)
            # Merge and append to main vector
            line_interp = np.array([x_interp,y_interp])
            self.unfolding_points = np.hstack((self.unfolding_points,line_interp))


    def unfold_image(self):
        """Creates an unfolded image of the honeycomb structure 
        using previously defined unfolding points.
        """
        if self.unfolding_points.shape[1] == 0:
            raise Exception('Unfolding points are not defined')

        x = np.linspace(0,self.img_shape[0]-1,self.img_shape[0])
        y = np.linspace(0,self.img_shape[1]-1,self.img_shape[1])
        img_interp = scipy.interpolate.RegularGridInterpolator((x, y), self.img, method='linear')
        temp_unf_points = np.swapaxes(self.unfolding_points,0,1)
        unfolded_img = img_interp(temp_unf_points)
        unfolded_img = unfolded_img.reshape((self.normals.shape[2],self.interp_points)).transpose()

        if self.visualize == True:
            plt.figure()
            plt.imshow(unfolded_img,cmap='gray')
            plt.show()

        return unfolded_img.astype(np.int32)
        

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
                point = unfolding_points_mat[:,j,surface[j]]
                # Append to surface vec
                folded_surface = np.hstack((folded_surface,np.expand_dims(point,axis=1)))
            # Sort surface ascending in relation to x axis
            folded_surfaces.append(folded_surface)
        return folded_surfaces




class HoneycombUnfoldSlicewise3d:
    lines = []
    lines_interp = np.empty((2,0))
    normals = np.empty((2,3,0))
    unfolding_points = np.empty((2,0))

    interp_points = 0
    interp_x_len = 0
    interp_z_len = 0

    def __init__(self, img, vis_img, visualize=True):
        self.img = img
        self.vis_img = vis_img
        self.img_shape = img.shape
        self.visualize = visualize
        self.slice_idx = [0, int(self.img_shape[0]/2), self.img_shape[0]-1]


    def draw_corners(self):

        sliceList = ["Bottom", "Middle", "Top"]
        # Loop through the slices (top, middle, bottom)
        for i in range(3):
            # Create image for pointing the corners
            plt.figure()
            plt.imshow(self.vis_img[self.slice_idx[i],:,:], cmap='gray')
            plt.suptitle(f"{sliceList[i]} slice: Click on the corners of folds of the chosen unmarked layer from left to right.")
            plt.title("Left button - new point, Right button - remove point, Middle button - end process", fontsize=8)

            ax = plt.gca()
            xy = plt.ginput(-1, timeout=60)
            plt.close()
            # Get point coordinates and append to a list
            x = [p[0] for p in xy]
            y = [p[1] for p in xy]
            z = [self.slice_idx[i] for j in range(len(x))]
            line = np.array([x,y,z])
            self.lines.append(line)
            # Draw the lines onto the visualization image (needed for multiple surface caluclation)
            for j in range(line.shape[1]-1):
                start_point = (int(np.round(line[0,j])), int(np.round(line[1,j])))
                end_point = (int(np.round(line[0,j+1])), int(np.round(line[1,j+1])))
                color = (0, 0, 255)
                thickness = 9
                self.vis_img[self.slice_idx[i],:,:] = cv.line(self.vis_img[self.slice_idx[i],:,:], start_point, end_point, color, thickness)
        # Plot the results of the corner drawing
        if self.visualize == True:
            plt.figure()
            for i in range(3):
                line = self.lines[i]
                plt.subplot(1,3,i+1)
                plt.imshow(self.img[self.slice_idx[i],:,:], cmap='gray')
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
                zlayer = np.argmin(np.abs(zUnique-self.slice_idx[i]))
                # Get points for the layer
                interp_points = self.lines_interp[0:2,zlayer*self.interp_x_len:(zlayer+1)*self.interp_x_len]

                plt.subplot(1,3,i+1)
                plt.imshow(self.img[self.slice_idx[i],:,:], cmap='gray')
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
                zlayer = np.argmin(np.abs(zUnique-self.slice_idx[i]))
                # Get points for the layer
                interp_points = self.lines_interp[0:2,zlayer*self.interp_x_len:(zlayer+1)*self.interp_x_len]

                plt.subplot(1,3,i+1)
                plt.imshow(self.img[self.slice_idx[i],:,:], cmap='gray')
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
                zlayer = np.argmin(np.abs(zUnique-self.slice_idx[i]))
                # Get normals for the layer
                temp_normals = self.normals[:,0:2,zlayer*self.interp_x_len:(zlayer+1)*self.interp_x_len]

                plt.subplot(1,3,i+1)
                plt.imshow(self.img[self.slice_idx[i],:,:], cmap='gray')
                for j in range(temp_normals.shape[2]):
                    plt.plot(temp_normals[:,0,j],temp_normals[:,1,j], '-',color='k')
            plt.show()