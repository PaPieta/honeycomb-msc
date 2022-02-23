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
    """Set of methods for unfolding and folding a honeycomb scan
    for purposes of slicewise layered surfaces detection of a 3D stack.\n
    Pipeline of the segmentation process:\n
    draw_corners -> interpolate_points -> smooth_interp-corners ->
    calculate_normals -> get_unfold_points_from_normals -> unfold_image ->
    (external) segment layered surfaces -> fold_surfaces_back"""
    

    def __init__(self, img, vis_img, visualize=True):
        """Class initialization. 
        Params:\n
        img - 3D image stack of the honeycmb scan\n
        vis_img - deep copy of the image stack for visualization purposes\n
        visualize - if True, will visualize each step of the unfolding process\n
        Attributes:\n
        img, vis_img, visualize - see above\n
        slice_idx - 3 element list of slice indices representing first, central and last slice\n
        lines - list of corner points marked on the image representing the lines along the honeycomb structure\n
        lines_interp - array of points interpolated along the lines
        normals - array of lines normal to the orignal structure line, calculated for each interpolated point\n
        unfolding_points - points interpolated along the normal lines, used for unfolding of the image\n
        interp_points - number of interpolation points along each normal line\n
        interp_x_len, interp_z_len - size of the interpolation matrix corelated to image X and Z axis\n
        """
        self.img = img
        self.vis_img = vis_img
        self.img_shape = img.shape
        self.visualize = visualize
        self.slice_idx = [0, int(self.img_shape[0]/2), self.img_shape[0]-1]

        self.lines = []
        self.lines_interp = np.empty((2,0))
        self.normals = np.empty((2,3,0))
        self.unfolding_points = np.empty((3,0))

        self.interp_points = 0
        self.interp_x_len = 0
        self.interp_z_len = 0


    def draw_corners(self):
        """Opens interactive plot on a provided image to allow user to draw
        corners of a chosen honeycomb surface on 3 Z stack levels.
        """
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
        """interpolates points along marked corners on all Z stack layers with a chosen step.\n
        Params:\n
        step - interpolation step distance in the XY slice
        """
        if len(self.lines) == 0:
            raise Exception('No lines are defined. Define the line first with draw_line')

        #For visualization
        zUnique = np.arange(0,self.img_shape[0],1) # !!!!!!!Warning changed Z step to be always 1
        # TODO: Think about possibly moving this function here
        interpMat = helpers.getLinesUniformInterpSpacing(self.lines, zStep=1, xStep=step)

        self.lines_interp = np.moveaxis(interpMat,1,-1).reshape(3,-1)

        # # save for calculating normals
        self.interp_x_len = interpMat.shape[1]
        self.interp_z_len = interpMat.shape[2]


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
        """Applies smoothing on the interpolation points along X and Y axis 
        to provide a smoother transition in the corners of the strucuture.
        """
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
        """Calculate lines normal to the interpolated points with given range.\n
        Params:\n
        normals_range - length range of each normal line (length = 2*range)
        """
        if self.lines_interp.size == 0:
            raise Exception('Interpolated points are not defined')

        # Get matrix from the list
        linesInterpMat = np.reshape(self.lines_interp,(3,self.interp_z_len,self.interp_x_len))
        # Take only x y values
        linesInterpMat2d = linesInterpMat[:2,:,:]
        # Create matrix of vectors
        vecMat = np.zeros_like(linesInterpMat2d)
        # Fill central values
        vecMat[:,:,1:-1] = linesInterpMat2d[:,:,2:] - linesInterpMat2d[:,:,:-2]
        # Fill first and last column 
        vecMat[:,:,0] = linesInterpMat2d[:,:,1] - linesInterpMat2d[:,:,0]
        vecMat[:,:,-1] = linesInterpMat2d[:,:,-1] - linesInterpMat2d[:,:,-2]
        # Normalize
        vecMat = vecMat/np.apply_along_axis(np.linalg.norm,0,vecMat)
        # Calculate perpendicular vector
        perVecMat = np.empty_like(vecMat)
        perVecMat[0,:,:] = -vecMat[1,:,:]
        perVecMat[1,:,:] = vecMat[0,:,:]
        # Calculate points movec by vector in both directions
        normP1 = linesInterpMat2d + perVecMat*normals_range
        normP2 = linesInterpMat2d - perVecMat*normals_range
        # Include z axis position
        normP1 = np.vstack((normP1,np.array([linesInterpMat[2,:,:]])))
        normP2 = np.vstack((normP2,np.array([linesInterpMat[2,:,:]])))
        # Combine points
        normalLines = np.array([normP1,normP2])
        self.normals = np.reshape(normalLines,(2,3,-1))

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
            z1 = self.normals[0,2,i]
            z2 = self.normals[1,2,i]
            # Interpolate x, y, z
            line_interp = np.linspace([x1,y1,z1],[x2,y2,z2],interp_points).transpose()
            self.unfolding_points = np.hstack((self.unfolding_points,line_interp))


    def get_unfold_points_from_normals2(self, interp_points=60):
        """Interpolates points along each normal line to create unfolding points.
        Speeds up the process by calculating all points in one Z axis column at once\n
        Params:\n
        interp_points - number of points interpolated along each normal line.
        """
        ### YXZ order of axes
        if self.normals.shape[2] == 0:
            raise Exception('normals are not defined')

        self.interp_points = interp_points

        normalsMat = np.reshape(self.normals,(2,3,self.interp_z_len,self.interp_x_len))
        temp_unf_points = np.zeros((3,self.normals.shape[2]*interp_points))
        for i in range(normalsMat.shape[3]):
            surf = normalsMat[:,:,:,i]
            surfFlat = np.reshape(np.moveaxis(surf,0,-1),(3,-1))
            # halfSurf1 = fullSurf[:,:,:self.slice_idx[1]]
            # halfSurf2 = fullSurf[:,:,self.slice_idx[1]+1:]
            # center = fullSurf[:,:,self.slice_idx[1]]
            
            zRow = np.array([np.linspace(surf[0,2,0],surf[1,2,0],interp_points)])
            zCol = np.array([np.arange(0,surf.shape[2],surf[0,2,1]-surf[0,2,0])])
            zArr = (zRow+1).transpose()@zCol
            zArr = zArr.transpose().ravel()
            xArr = np.zeros(surf.shape[2]*interp_points)
            
            for j in range(surf.shape[2]):
                xRow = np.linspace(surf[0,1,j]-0.05,surf[1,1,j]+0.05,interp_points)
                xArr[j*interp_points:(j+1)*interp_points] = xRow
            xi = np.array((xArr,zArr)).transpose()
            points = np.array((surfFlat[1, :],surfFlat[2, :])).transpose()
            # create interpolation function  x,z,y order
            yArr = scipy.interpolate.griddata(points,surfFlat[0, :], xi)
            halfInterp = np.array((xArr,yArr,zArr))
            # self.unfolding_points = np.hstack((self.unfolding_points,halfInterp))
            temp_unf_points[:,i*surf.shape[2]*interp_points:(i+1)*surf.shape[2]*interp_points] = halfInterp

        unf_reshaped = np.reshape(temp_unf_points,(3,self.interp_x_len,self.interp_z_len,interp_points))
        unf_reshaped = np.reshape(np.swapaxes(unf_reshaped,1,2),(3,-1))

        self.unfolding_points = unf_reshaped


    def unfold_image(self):
        """Creates an unfolded image of the honeycomb structure 
        using previously defined unfolding points.
        """
        if self.unfolding_points.shape[1] == 0:
            raise Exception('Unfolding points are not defined')

        x = np.linspace(0,self.img_shape[1]-1,self.img_shape[1])
        y = np.linspace(0,self.img_shape[2]-1,self.img_shape[2])
        z = np.linspace(0,self.img_shape[0]-1,self.img_shape[0])
        img_interp = scipy.interpolate.RegularGridInterpolator((z, x, y), self.img, method='linear')
        temp_unf_points = np.swapaxes(self.unfolding_points,0,1)[:,[2,0,1]]
        unfolded_img = img_interp(temp_unf_points)
        unfolded_img = unfolded_img.reshape((self.interp_z_len, self.interp_x_len,self.interp_points))
        unfolded_img = np.swapaxes(unfolded_img, 1, 2)

        if self.visualize == True:
            zUnique = np.unique(self.lines_interp[2,:])
            plt.figure()
            for i in range(3):
                # Find interpolated z layer closest to the original image layers
                zlayer = np.argmin(np.abs(zUnique-self.slice_idx[i]))

                plt.subplot(3,1,i+1)
                plt.imshow(unfolded_img[zlayer,:,:], cmap='gray')
            plt.show()

        return unfolded_img.astype(np.int32)


    def fold_surfaces_back(self, surfaces, zIdx):
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
        unfolding_points_mat = temp_unf_points.reshape(2,self.interp_z_len,self.interp_x_len,self.interp_points)
        folded_surfaces = []
        for i in range(len(surfaces)):
            folded_surface = np.empty((2,0))
            surface = surfaces[i]
            for j in range(surface.shape[0]):
                # Get point position on original image
                if surface[j].is_integer():
                    point = unfolding_points_mat[:,zIdx,j,int(surface[j])]
                else:
                    surfVal = surface[j]
                    #Edge case
                    if unfolding_points_mat.shape[3] - surfVal < 1:
                        p1 = unfolding_points_mat[:,zIdx,j,int(np.floor(surfVal))]
                    else:
                        p1 = unfolding_points_mat[:,zIdx,j,int(np.ceil(surfVal))]
                    p2 = unfolding_points_mat[:,zIdx,j,int(np.floor(surfVal))]
                    point = p1*(np.ceil(surfVal)-surfVal) + p2*(surfVal-np.floor(surfVal))
                # Append to surface vec
                folded_surface = np.hstack((folded_surface,np.expand_dims(point,axis=1)))
            folded_surfaces.append(folded_surface)
        return folded_surfaces