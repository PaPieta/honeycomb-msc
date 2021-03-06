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


# from mpl_toolkits import mplot3d
# from mayavi import mlab
# from scipy.spatial import Delaunay

from honeycomb.helpers import misc



class Unfold3d:
    """Set of methods for unfolding and folding a honeycomb scan
    for purposes of slicewise layered surfaces detection of a 3D stack.\n
    Pipeline of the segmentation process:\n
    draw_corners -> interpolate_points -> smooth_interp-corners ->
    calculate_normals -> get_unfold_points_from_normals -> unfold_image ->
    (external) segment layered surfaces -> fold_surfaces_back"""
    

    def __init__(self, img, vis_img, visualize=True):
        """Class initialization.n
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
        unfolded_img - 3D image stack of the unfolded image
        """
        self.img = img
        self.vis_img = vis_img
        self.img_shape = img.shape
        self.visualize = visualize
        self.slice_idx = [0, int(self.img_shape[0]/2), self.img_shape[0]-1]

        self.lines = []
        self.lines_interp = np.empty((3,0,0))
        self.normals = np.empty((2,3,0))
        self.unfolding_points = np.empty((3,0,0,0))

        self.interp_points = 0
        self.interp_x_len = 0
        self.interp_z_len = 0

        self.unfolded_img = np.empty((1,0))


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
            plt.figure(figsize=(18,6))
            for i in range(3):
                line = self.lines[i]
                plt.subplot(1,3,i+1)
                plt.imshow(self.img[self.slice_idx[i],:,:], cmap='gray')
                plt.plot(line[0,:],line[1,:], '-')
                plt.title(f"Z={self.slice_idx[i]}")
            plt.suptitle("Marked points")
            plt.show()
            # pointVec = np.empty((3,0))
            # for i in range(3):
            #     pointVec = np.hstack((pointVec,self.lines[i]))
            # p2d = np.vstack([pointVec[0,:],pointVec[2,:]]).T
            # d2d = Delaunay(p2d)
            # fig = mlab.figure(1, bgcolor=(1, 1, 1),fgcolor=(0.,0.,0.))
            # tmesh = mlab.triangular_mesh(pointVec[0,:], pointVec[2,:],pointVec[1,:], d2d.vertices,representation='wireframe', color=(0,0,0))
            # mlab.axes(color=(0,0,0),extent=(0,int(np.max(pointVec[0,:])),0,int(np.max(pointVec[2,:])),0,int(np.max(pointVec[1,:]))),
            #     xlabel='X',ylabel='Z',zlabel='Y')
            # mlab.xlabel("X", color=(0,0,0))
            # mlab.ylabel("Z", color=(0,0,0))
            # mlab.zlabel("Y", color=(0,0,0))
                
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
        self.lines_interp = misc.getLinesUniformInterpSpacing(self.lines, zStep=1, xStep=step)

        # # save for calculating normals
        self.interp_x_len = self.lines_interp.shape[1]
        self.interp_z_len = self.lines_interp.shape[2]


        if self.visualize == True:
            plt.figure(figsize=(18,6))
            for i in range(3):
                # Find interpolated z layer closest to the original image layers
                zlayer = np.argmin(np.abs(zUnique-self.slice_idx[i]))
                # Get points for the layer
                interp_points = self.lines_interp[0:2,:,zlayer]

                plt.subplot(1,3,i+1)
                plt.imshow(self.img[self.slice_idx[i],:,:], cmap='gray')
                plt.plot(interp_points[0,:],interp_points[1,:], '*',markersize=4)
                plt.title(f"Z={zlayer}")
            plt.suptitle("Interpolated points")
            plt.show()
    
    def smooth_interp_corners(self):
        """Applies smoothing on the interpolation points along X and Y axis 
        to provide a smoother transition in the corners of the strucuture.
        """
        if self.lines_interp.size == 0:
            raise Exception('Interpolated points are not defined')
        # Pass x and y through gaussian smoothing filter
        for i in range(self.interp_z_len):
            self.lines_interp[0,:,i] = scp.gaussian_filter1d(self.lines_interp[0,:,i],2)
            self.lines_interp[1,:,i] = scp.gaussian_filter1d(self.lines_interp[1,:,i],2)

        if self.visualize == True:
            zUnique = np.unique(self.lines_interp[2,:])
            plt.figure(figsize=(18,6))
            for i in range(3):
                # Find interpolated z layer closest to the original image layers
                zlayer = np.argmin(np.abs(zUnique-self.slice_idx[i]))
                # Get points for the layer
                interp_points = self.lines_interp[0:2,:,zlayer]

                plt.subplot(1,3,i+1)
                plt.imshow(self.img[self.slice_idx[i],:,:], cmap='gray')
                plt.plot(interp_points[0,:],interp_points[1,:], '*',markersize=1)
                plt.title(f"Z={self.slice_idx[i]}")
            plt.suptitle("Smoothed points")
            plt.show()
            # fig = mlab.figure(1, bgcolor=(1, 1, 1),fgcolor=(0.,0.,0.))
            # mlab.points3d(self.lines_interp[0,:,:],self.lines_interp[1,:,:],self.lines_interp[2,:,:],color=(1,0.2,0.2),scale_factor=3)
            # mlab.axes(color=(0,0,0),extent=(0,int(np.max(self.lines_interp[0,:,:])),0,int(np.max(self.lines_interp[1,:,:])),0,int(np.max(self.lines_interp[2,:,:]))),
            #     xlabel='X',ylabel='Y',zlabel='Z')


    def calculate_normals(self,normals_range=30):
        """Calculate lines normal to the interpolated points with given range.\n
        Params:\n
        normals_range - length range of each normal line (length = 2*range)
        """
        if self.lines_interp.size == 0:
            raise Exception('Interpolated points are not defined')

        # Take only x y values
        lines_interp_xy = self.lines_interp[:2,:,:]
        # Create matrix of vectors
        vecMat = np.zeros_like(lines_interp_xy)
        # Fill central values
        vecMat[:,1:-1,:] = lines_interp_xy[:,2:,:] - lines_interp_xy[:,:-2,:]
        # Fill first and last column 
        vecMat[:,0,:] = lines_interp_xy[:,1,:] - lines_interp_xy[:,0,:]
        vecMat[:,-1,:] = lines_interp_xy[:,-1,:] - lines_interp_xy[:,-2,:]
        # Normalize
        vecMat = vecMat/np.apply_along_axis(np.linalg.norm,0,vecMat)
        # Calculate perpendicular vector
        perVecMat = np.empty_like(vecMat)
        perVecMat[0,:,:] = -vecMat[1,:,:]
        perVecMat[1,:,:] = vecMat[0,:,:]
        # Calculate points moved by vector in both directions
        normP1 = lines_interp_xy - perVecMat*normals_range
        normP2 = lines_interp_xy + perVecMat*normals_range
        # Include z axis position
        normP1 = np.vstack((normP1,np.array([self.lines_interp[2,:,:]])))
        normP2 = np.vstack((normP2,np.array([self.lines_interp[2,:,:]])))
        # Combine points
        self.normals = np.array([normP1,normP2])

        if self.visualize == True:
            zUnique = np.unique(self.lines_interp[2,:])
            plt.figure(figsize=(18,6))
            for i in range(3):
                # Find interpolated z layer closest to the original image layers
                zlayer = np.argmin(np.abs(zUnique-self.slice_idx[i]))
                # Get normals for the layer
                temp_normals = self.normals[:,0:2,:,zlayer]

                plt.subplot(1,3,i+1)
                plt.imshow(self.img[self.slice_idx[i],:,:], cmap='gray')
                for j in range(temp_normals.shape[2]):
                    plt.plot(temp_normals[:,0,j],temp_normals[:,1,j], '-',color='k')
                    plt.title(f"Z={self.slice_idx[i]}")
            plt.suptitle("Calculated normal lines")
            plt.show()

    def get_unfold_points_from_normals(self, interp_points=60):
        """Interpolates points along each normal line to create unfolding points.
        Speeds up the process by calculating all points in one Z axis column at once\n
        Params:\n
        interp_points - number of points interpolated along each normal line.
        """
        if self.normals.shape[2] == 0:
            raise Exception('normals are not defined')

        self.interp_points = interp_points

        # Z, Y, X shape in order to fit original image definition
        self.unfolding_points = np.zeros((3,self.normals.shape[3],interp_points,self.normals.shape[2]))
        for i in range(self.normals.shape[2]):
            surf = self.normals[:,:,i,:]
            surfFlat = np.reshape(np.moveaxis(surf,0,-1),(3,-1))
            
            zRow = np.array([np.linspace(surf[0,2,0],surf[1,2,0],interp_points)])
            zCol = np.array([np.arange(0,surf.shape[2],surf[0,2,1]-surf[0,2,0])])
            zArr = (zRow+1).transpose()@zCol
            zArr = zArr.ravel()
            yArr = np.zeros((interp_points,surf.shape[2]))
            
            for j in range(surf.shape[2]):
                y1 = surf[0,1,j]
                y2 = surf[1,1,j]
                absdiff = abs(y1-y2)
                if y1<y2: 
                    y1 = y1 + absdiff/1000
                    y2 = y2 - absdiff/1000
                elif y1>y2:
                    y1 = y1 - absdiff/1000
                    y2 = y2 + absdiff/1000
                yArr[:,j] = np.linspace(y1,y2,interp_points)
            yArr = yArr.ravel()
            xi = np.array((yArr,zArr)).transpose()
            points = np.array((surfFlat[1, :],surfFlat[2, :])).transpose()
            # create interpolation function  x,z,y order
            xArr = scipy.interpolate.griddata(points,surfFlat[0, :], xi)
            interpSurf = np.array((xArr,yArr,zArr)).reshape(3,interp_points,zCol.shape[1])

            # Swap from YZ to ZY and assign
            self.unfolding_points[:,:,:,i] = np.swapaxes(interpSurf,1,2)


    def unfold_image(self, method='linear'):
        """Creates an unfolded image of the honeycomb structure 
        using previously defined unfolding points.
        Params:\n
        method - interpolation method, 'lienar' or 'nearest'
        """
        if self.unfolding_points.shape[1] == 0:
            raise Exception('Unfolding points are not defined')

        y = np.linspace(0,self.img_shape[1]-1,self.img_shape[1])
        x = np.linspace(0,self.img_shape[2]-1,self.img_shape[2])
        z = np.linspace(0,self.img_shape[0]-1,self.img_shape[0])
        img_interp = scipy.interpolate.RegularGridInterpolator((z, y, x), self.img, method=method)
        # Flatten unfolding points to 3,-1, swap to -1,3 and change order from xyz to zyx
        temp_unf_points = np.swapaxes(self.unfolding_points.reshape(3,-1),0,1)[:,[2,1,0]]

        unfolded_img = img_interp(temp_unf_points)
        unfolded_img = unfolded_img.reshape((self.interp_z_len, self.interp_points, self.interp_x_len))
        
        self.unfolded_img = unfolded_img.astype(np.int32)

        if self.visualize == True:
            zUnique = np.unique(self.lines_interp[2,:])
            plt.figure(figsize=(18,6))
            for i in range(3):
                # Find interpolated z layer closest to the original image layers
                zlayer = np.argmin(np.abs(zUnique-self.slice_idx[i]))

                plt.subplot(3,1,i+1)
                plt.imshow(unfolded_img[zlayer,:,:], cmap='gray')
            plt.show()

        return self.unfolded_img


    def fold_2d_surfaces_back(self, surfaces, zIdx):
        """Folds 2D surfaces back to original shape. Returns a line with original data points\n
        Params:\n
        surfaces - list of surfaces to unfold\n
        zIdx - zAxis index of the surface
        """
        if self.unfolding_points.shape[1] == 0:
            raise Exception('Unfolding points are not defined')
        # Create "coordinate image" out of normals unfolding points
        unfolding_points_yx = self.unfolding_points[[0,1],:,:,:]
        folded_surfaces = []
        for i in range(len(surfaces)):
            folded_surface = np.empty((2,0))
            surface = surfaces[i]
            for j in range(surface.shape[0]):
                # Get point position on original image
                if surface[j].is_integer():
                    point = unfolding_points_yx[:,zIdx,int(surface[j]),j]
                else:
                    surfVal = surface[j]
                    #Edge case
                    if unfolding_points_yx.shape[2] - surfVal < 1:
                        p1 = unfolding_points_yx[:,zIdx,int(np.floor(surfVal)),j]
                    else:
                        p1 = unfolding_points_yx[:,zIdx,int(np.ceil(surfVal)),j]
                    p2 = unfolding_points_yx[:,zIdx,int(np.floor(surfVal)),j]
                    point = p1*(np.ceil(surfVal)-surfVal) + p2*(surfVal-np.floor(surfVal))
                # Append to surface vec
                folded_surface = np.hstack((folded_surface,np.expand_dims(point,axis=1)))
            folded_surfaces.append(folded_surface)
        return folded_surfaces

    def fold_3d_surfaces_back(self, surfaces, representation='vector'):
        """Folds 3D surfaces back to original shape. Returns a line with original data points\n
        Params:\n
        surfaces - list of surfaces to unfold\n
        representation - 'vector' for (3,z*x) vector, or 'matrix' for (3,z,x) matrix
        """
        if self.unfolding_points.shape[1] == 0:
            raise Exception('Unfolding points are not defined')
        # # Create "coordinate image" out of normals unfolding points
        # unfolding_points_yx = self.unfolding_points[[0,1],:,:,:]
        folded_surfaces = []
        for i in range(len(surfaces)):
            surface = surfaces[i]
            if representation == 'vector':
                folded_surface = np.empty((3,surface.shape[0]*surface.shape[1]))
            elif representation == 'matrix':
                folded_surface = np.empty((3,surface.shape[0],surface.shape[1]))
            else:
                raise AttributeError("Wrong representation type, has to be either 'vector' or 'matrix'")
            for j in range(surface.shape[0]):
                for k in range(surface.shape[1]):
                    # Get point position on original image
                    if surface[j,k].is_integer():
                        point = self.unfolding_points[:,j,int(surface[j,k]),k]
                    else:
                        surfVal = surface[j,k]
                        #Edge case
                        if self.unfolding_points.shape[2] - surfVal < 1:
                            p1 = self.unfolding_points[:,j,int(np.floor(surfVal)),k]
                        else:
                            p1 = self.unfolding_points[:,j,int(np.ceil(surfVal)),k]
                        p2 = self.unfolding_points[:,j,int(np.floor(surfVal)),k]
                        point = p1*(np.ceil(surfVal)-surfVal) + p2*(surfVal-np.floor(surfVal))
                    # Add to surface vec
                    if representation == 'vector':
                        # folded_surface = np.hstack((folded_surface,np.expand_dims(point,axis=1)))
                        folded_surface[:,(j*surface.shape[1])+k] = point
                    else:
                        folded_surface[:,j,k] = point
            folded_surfaces.append(folded_surface)
        return folded_surfaces