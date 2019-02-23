#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 traditional Approaches

Author(s): 
Darshan Shah (dshah003@umd.edu)
M.Eng. in Robotics,
University of Maryland, College Park

Mayank Pathak (pathak10@umd.edu)
M.Eng. in Robotics,
University of Maryland, College Park
"""

# Code starts here:

import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt, numpy as np
import argparse

def main():
	# Add any Command Line arguments here
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

	Args = Parser.parse_args()
	NumFeatures = Args.NumFeatures
	"""
	Read a set of images for Panorama stitching
	"""
	I1 = cv2.imread('../Data/Train/Set1/2.jpg')
	I2 = cv2.imread('../Data/Train/Set1/3.jpg')

	gray1 = cv2.cvtColor(I1 ,cv2.COLOR_RGB2GRAY)
	gray2 = cv2.cvtColor(I2 ,cv2.COLOR_RGB2GRAY)
	gray1 = np.float32(gray1)
	gray2 = np.float32(gray2)

	gray1 = cv2.medianBlur(gray1,5)
	gray2 = cv2.medianBlur(gray2,5)


	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""
	corner1 = cv2.goodFeaturesToTrack(gray1, NumFeatures, 0.01, 10, 3 )
	corner1 = np.int0(corner1)

	corner2 = cv2.goodFeaturesToTrack(gray2, NumFeatures,0.01, 10, 3 )
	corner2 = np.int0(corner2)

	corner1 = corner1.reshape(NumFeatures,-1)
	corner2 = corner2.reshape(NumFeatures,-1)

	# Draw Corners on Image and Save.
	I1_copy = I1.copy()
	I2_copy = I2.copy()

	red = [0,0,255]
	for i in range(0,len(corner1)):
		cv2.circle(I1_copy,(corner1[i,0], corner1[i,1]),10,red,-1)
		cv2.circle(I2_copy,(corner2[i,0], corner2[i,1]),10,red,-1)

	cv2.imwrite("I1_corners.png", I1_copy)
	cv2.imwrite("I2_corners.png", I2_copy)


	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

	Feature1 = extract(gray1, corner1)
	Feature2 = extract(gray2, corner2)


	"""
	Feature Matching
	Save Feature Matching output as matching.png
	"""
	matches = matching(Feature1, Feature2)

	# index1 = matches[:,0]
	# index2 = matches[:,1]
	Match1 = []
	Match2 = []
	Match1.append(corner1[matches[:,0]])
	Match2.append(corner2[matches[:,1]])
	Match1 = np.array(Match1)
	Match2 = np.array(Match2)
	Match1 = Match1.reshape(Match1.shape[1],2)
	Match2 = Match2.reshape(Match2.shape[1],2)

	Out = drawMatches(gray1, Match1, gray2, Match2)

	"""
	Refine: RANSAC, Estimate Homography
	"""
	Match1 = np.array(Match1, dtype=np.float)
	Match2 = np.array(Match2, dtype=np.float)

	M, mask = cv2.findHomography(Match1, Match2, cv2.RANSAC, 10)


	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
	matchesMask = mask.ravel().tolist()
	GoodKeypoints1 = []
	GoodKeypoints2 = []
	for i in range(len(matchesMask)):
		if(matchesMask[i] != 0):
			GoodKeypoints1.append(Match1[i])
			GoodKeypoints2.append(Match2[i])
	GoodKeypoints1 = np.array(GoodKeypoints1, dtype=np.int)
	GoodKeypoints2 = np.array(GoodKeypoints2, dtype=np.int)
	Out = drawMatches(gray1, GoodKeypoints1, gray2, GoodKeypoints2)

	result = warpTwoImages(I2, I1, M)
	cv2.imwrite("mypano.png", result)


def matching(d1, d2):
    h, w, n = d1.shape[0:3]
    ds = cdist((d1.reshape((w**2, n))).T, (d2.reshape((w**2, n))).T)
    bt = np.argsort(ds, 1)[:, 0]
    ratio = ds[np.r_[0:n], bt] / ds[np.r_[0:n], np.argsort(ds, 1)[:, 1]].mean()
    return np.hstack([np.argwhere(ratio < 0.5), bt[np.argwhere(ratio < 0.5)]]).astype(int)

def extract(img, harris, radius=8):
    # Change number here to change the scale. 4 is the optimal amount 
    y, x = 4 * np.mgrid[-radius:radius+1, -radius:radius+1]
    desc = np.zeros((2 * radius + 1, 2 * radius + 1, harris.shape[0]), dtype=float)
    for i in range(harris.shape[0]):
        patch = map_coordinates(img,[harris[i,1] + y, harris[i,0] + x], prefilter=False)
        desc[..., i] = (patch - patch.mean()) / patch.std()
    return desc

def drawMatches(Img1, Kp1, Img2, Kp2):
    h1, w1 = Img1.shape
    h2, w2 = Img2.shape
    Outimg = np.zeros([max(h1, h2), (w1+w2)])
    Outimg[0:h1, 0:w1] = Img1
    Outimg[0:h2, w1:w1+w2] = Img2
    for i in range(len(Kp1)):
        x1, y1 = Kp1[i]
        x2, y2 = Kp2[i]
        cv2.circle(Outimg, (x1, y1), 10, (0))
        cv2.circle(Outimg, (x2+w1, y2), 7, (255))
        cv2.line(Outimg, (x1,y1), (x2+w1, y2), (255), thickness = 7, lineType = 8)
    plt.figure(figsize=(17, 10))
    print(type(Outimg))
    plt.imshow(Outimg, cmap = 'gray')

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result





if __name__ == '__main__':
    main()
 
