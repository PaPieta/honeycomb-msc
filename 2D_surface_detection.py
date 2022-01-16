# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 17:10:52 2022

@author: pawel
"""

import numpy as np 
import matplotlib.pyplot as plt
import skimage.io 
import scipy.ndimage as scp
import scipy
import slgbuilder


def rescaleImage(img, minVal, maxVal):
    img = ((img - img.min()) * (minVal/(img.max() - img.min()) * maxVal)).astype(np.int32)
    return img

class HoneycombUnfold:
    lines = np.array([])
    lines_interp = np.empty((2,0))
    normals = np.empty((2,2,0))
    unfolding_points = np.empty((2,0))

    interp_points=0

    def __init__(self, img, visualize=True):
        self.img = img
        self.img_shape = img.shape
        self.visualize = visualize

        if visualize == True:
            self.fig, self.ax = plt.subplots(2,2)

    def draw_corners(self):

        plt.figure()
        plt.imshow(self.img, cmap='gray')
        plt.suptitle("Click on the corners of one of the folds from left to right.")
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

    def interpolate_points(self,num_points=50):
        if self.lines.size == 0 or self.lines.shape[1] < 2:
            raise Exception('No lines are defined. Define the line first with draw_line')
        for i in range(self.lines.shape[1]-1):
            # Get linterpolation start and end points
            x1 = self.lines[0,i]
            x2 = self.lines[0,i+1]
            y1 = self.lines[1,i]
            y2 = self.lines[1,i+1]
            # Interpolate x and y
            x_interp = np.linspace(x1,x2,num_points)
            y_interp = np.linspace(y1,y2,num_points)
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
        
    def fold_lines_back(self, lines, interpolate=True):
        """Folds line back to original shape. Returns a line with original data points 
        or with one (rounded) value for each x axis pixel, if interpolate=True"""
        # TODO: better interpolation
        if self.unfolding_points.shape[1] == 0:
            raise Exception('Unfolding points are not defined')
        # Create "coordinate image" out of normals unfolding points
        unfolding_points_mat = self.unfolding_points.reshape(2,self.normals.shape[2],self.interp_points)
        folded_lines = []
        for i in range(len(lines)):
            folded_line = np.empty((2,0))
            line = lines[i]
            for j in range(line.shape[0]):
                # Get point position on original image
                point = unfolding_points_mat[:,j,line[j]]
                # Append to line vec
                folded_line = np.hstack((folded_line,np.expand_dims(point,axis=1)))
            # Sort line ascending in relation to x axis
            folded_line = folded_line[:,np.argsort(folded_line[0, :])]
            if interpolate == True:
                # Create interpolated line
                f = scipy.interpolate.interp1d(folded_line[0, :], folded_line[1, :])
                xnew = np.arange(np.ceil(folded_line[0, 0]),np.floor(folded_line[0, -1]),1)
                ynew = np.round(f(xnew))
                folded_line = np.array([xnew,ynew])

            folded_lines.append(folded_line)
        return folded_lines


I = skimage.io.imread('data/29-2016_29-2016-60kV-zoom-center_recon.tif')

I_2d = I[500,:,:]
# Median filter for removing salt and pepper-like noise
I_2d = scp.median_filter(I_2d, size=5)

hc = HoneycombUnfold(I_2d, visualize=False)
hc.draw_corners() 
hc.interpolate_points()
hc.smooth_interp_corners()
hc.calculate_normals(normals_range=30)
hc.get_unfold_points_from_normals(interp_points=60)
unfolded_img = hc.unfold_image()

# Rescale image - the black background pixels are removed
unfolded_img = rescaleImage(unfolded_img, 1, 255)

# plt.figure()
# plt.imshow(unfolded_img, cmap='gray')
# plt.show()

scale = 50
layers = [slgbuilder.GraphObject(0*unfolded_img), slgbuilder.GraphObject(0*unfolded_img), # no on-surface cost
            slgbuilder.GraphObject(scale*unfolded_img), slgbuilder.GraphObject(scale*unfolded_img)] # extra 2 dark lines
helper = slgbuilder.MaxflowBuilder()
helper.add_objects(layers)

# Adding regional costs, 
# the region in the middle is bright compared to two darker regions.
helper.add_layered_region_cost(layers[0], unfolded_img, np.max(unfolded_img)-unfolded_img)
helper.add_layered_region_cost(layers[1], np.max(unfolded_img)-unfolded_img, unfolded_img)

# Adding geometric constrains
helper.add_layered_boundary_cost() # blocks crossing from bottom to top of the image
helper.add_layered_smoothness(delta=2, wrap=False)  # line smoothness term
helper.add_layered_containment(layers[0], layers[1], min_margin=4, max_margin=25) # honeycomb lines pair
helper.add_layered_containment(layers[2], layers[3], min_margin=10, max_margin=45) # dark helper lines
helper.add_layered_containment(layers[2], layers[0], min_margin=1) # top dark line and top honeycomb line
helper.add_layered_containment(layers[1], layers[3], min_margin=1) # bottom honeycomb line and bottom dark line

# Cut
helper.solve()
segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
segmentation_lines = [s.shape[0] - np.argmax(s[::-1,:], axis=0) - 1 for s in segmentations]

# Visualization
plt.figure()
plt.imshow(unfolded_img, cmap='gray')
for i in range(len(segmentation_lines)):
    if i < 2:
        plt.plot(segmentation_lines[i], 'r')
    else:
        plt.plot(segmentation_lines[i], 'b')

plt.show()


folded_lines = hc.fold_lines_back(segmentation_lines,interpolate=True)

plt.figure()
plt.imshow(I_2d, cmap='gray')
for i in range(len(folded_lines)):
    folded_line = folded_lines[i]
    if i < 2:
        plt.plot(folded_line[0,:],folded_line[1,:], 'r')
    else:
        plt.plot(folded_line[0,:],folded_line[1,:], 'b')

plt.show()