"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

from __future__ import print_function, absolute_import, division
import tensorflow as tf
import sys
import numpy as np
#import cv2
#import os
#import glob
#from utils import utils
#from Misc.utils import utils.py
#from Misc.utils import *
#import tensorflow.contrib.slim as slim
#from utils.utils import *
#from utils.tf_spatial_transformer import transformer
#import pdb
#from tensorflow.contrib.layers.python.layers import initializers
#from collections import namedtuple
#from utils.utils import get_symetric_census, get_batch_symetric_census
# Don't generate pyc codes
sys.dont_write_bytecode = True

def HomographyModel(Img, ImageSize, MiniBatchSize, keep_prob):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Keep_prob - probality for dropout
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
#     Image_size = 
    #############################
    # Fill your network here!
    #############################
   
#     def network(Img,keep_prob):
    # conv1 conv2 maxpooling
    print("Entering Network\n\n")
    w1 = tf.Variable(tf.truncated_normal([3,3,2,64], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv1 = tf.nn.conv2d(Img, w1, strides=[1,1,1,1], padding='SAME') + b1
    mean1, var1 = tf.nn.moments(conv1, axes=[0, 1, 2])
    offset1 = tf.Variable(tf.constant(0.0, shape=[64]))
    scale1 = tf.Variable(tf.constant(1.0, shape=[64]))
    bn1 = tf.nn.batch_normalization(conv1, mean=mean1, variance=var1, offset=offset1, scale=scale1, variance_epsilon=1e-5)
    relu1 = tf.nn.relu(bn1)

    # print("reached P1")

    w2 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv2 = tf.nn.conv2d(relu1, w2, strides=[1,1,1,1], padding='SAME') + b2
    mean2, var2 = tf.nn.moments(conv2, axes=[0, 1, 2])
    offset2 = tf.Variable(tf.constant(0.0, shape=[64]))
    scale2 = tf.Variable(tf.constant(1.0, shape=[64]))
    bn2 = tf.nn.batch_normalization(conv2, mean=mean2, variance=var2, offset=offset2, scale=scale2, variance_epsilon=1e-5)
    relu2 = tf.nn.relu(bn2)
    maxpool1 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')

    # conv3 conv4 maxpooling
    w3 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv3 = tf.nn.conv2d(maxpool1, w3, strides=[1,1,1,1], padding='SAME') + b3
    mean3, var3 = tf.nn.moments(conv3, axes=[0, 1, 2])
    offset3 = tf.Variable(tf.constant(0.0, shape=[64]))
    scale3 = tf.Variable(tf.constant(1.0, shape=[64]))
    bn3 = tf.nn.batch_normalization(conv3, mean=mean3, variance=var3, offset=offset3, scale=scale3, variance_epsilon=1e-5)
    relu3 = tf.nn.relu(bn3)

    w4 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.1))
    b4 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv4 = tf.nn.conv2d(relu3, w4, strides=[1,1,1,1], padding='SAME') + b4
    mean4, var4 = tf.nn.moments(conv4, axes=[0, 1, 2])
    offset4 = tf.Variable(tf.constant(0.0, shape=[64]))
    scale4 = tf.Variable(tf.constant(1.0, shape=[64]))
    bn4 = tf.nn.batch_normalization(conv4, mean=mean4, variance=var4, offset=offset4, scale=scale4, variance_epsilon=1e-5)
    relu4 = tf.nn.relu(bn4)
    maxpool2 = tf.nn.max_pool(relu4, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')

    # print("Reached P2")
    # conv5 conv6 maxpooling
    w5 = tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1))
    b5 = tf.Variable(tf.constant(0.1, shape=[128]))
    conv5 = tf.nn.conv2d(maxpool2, w5, strides=[1,1,1,1], padding='SAME') + b5
    mean5, var5 = tf.nn.moments(conv5, axes=[0, 1, 2])
    offset5 = tf.Variable(tf.constant(0.0, shape=[128]))
    scale5 = tf.Variable(tf.constant(1.0, shape=[128]))
    bn5 = tf.nn.batch_normalization(conv5, mean=mean5, variance=var5, offset=offset5, scale=scale5, variance_epsilon=1e-5)
    relu5 = tf.nn.relu(bn5)

    w6 = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.1))
    b6 = tf.Variable(tf.constant(0.1, shape=[128]))
    conv6 = tf.nn.conv2d(relu5, w6, strides=[1,1,1,1], padding='SAME') + b6
    mean6, var6 = tf.nn.moments(conv6, axes=[0, 1, 2])
    offset6 = tf.Variable(tf.constant(0.0, shape=[128]))
    scale6 = tf.Variable(tf.constant(1.0, shape=[128]))
    bn6 = tf.nn.batch_normalization(conv6, mean=mean6, variance=var6, offset=offset6, scale=scale6, variance_epsilon=1e-5)
    relu6 = tf.nn.relu(bn6)
    maxpool3 = tf.nn.max_pool(relu6, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')

    # conv7 conv8 maxpooling
    w7 = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.1))
    b7 = tf.Variable(tf.constant(0.1, shape=[128]))
    conv7 = tf.nn.conv2d(maxpool3, w7, strides=[1,1,1,1], padding='SAME') + b7
    mean7, var7 = tf.nn.moments(conv7, axes=[0, 1, 2])
    offset7 = tf.Variable(tf.constant(0.0, shape=[128]))
    scale7 = tf.Variable(tf.constant(1.0, shape=[128]))
    bn7 = tf.nn.batch_normalization(conv7, mean=mean7, variance=var7, offset=offset7, scale=scale7, variance_epsilon=1e-5)
    relu7 = tf.nn.relu(bn7)

    w8 = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.1))
    b8 = tf.Variable(tf.constant(0.1, shape=[128]))
    conv8 = tf.nn.conv2d(relu7, w8, strides=[1,1,1,1], padding='SAME') + b8
    mean8, var8 = tf.nn.moments(conv6, axes=[0, 1, 2])
    offset8 = tf.Variable(tf.constant(0.0, shape=[128]))
    scale8 = tf.Variable(tf.constant(1.0, shape=[128]))
    bn8 = tf.nn.batch_normalization(conv8, mean=mean8, variance=var8, offset=offset8, scale=scale8, variance_epsilon=1e-5)
    relu8 = tf.nn.relu(bn8)
    dropout1 = tf.nn.dropout(relu8, keep_prob)

    reshape1 = tf.reshape(dropout1, [-1, 32768])
    w_fc1 = tf.Variable(tf.truncated_normal([32768,1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    fc1 = tf.matmul(reshape1, w_fc1) + b_fc1
    dropout2 = tf.nn.dropout(fc1, keep_prob)

    w_fc2 = tf.Variable(tf.truncated_normal([1024,8], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[8]))
    H4Pt = tf.matmul(dropout2, w_fc2) + b_fc2
    mean_var = [mean1,var1,mean2,var2,mean3,var3,mean4,var4,mean5,var5,mean6,var6,mean7,var7,mean8,var8]
    # mean_var is used in test process
    
#     return out, mean_var
    print("Returning from Network")
    print('H4pt',H4Pt)

    return H4Pt


def solve_DLT(self):
# Auxiliary matrices used to solve DLT
    Aux_M1  = np.array([
              [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [ 1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [ 0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
              [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [ 0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
              [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [ 0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)
    
    
    Aux_M2  = np.array([
              [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [ 0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [ 0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
              [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [ 0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
              [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ]], dtype=np.float64)
    
    
    
    Aux_M3  = np.array([
              [0],
              [1],
              [0],
              [1],
              [0],
              [1],
              [0],
              [1]], dtype=np.float64)
    
    
    
    Aux_M4  = np.array([
              [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)
    
    
    Aux_M5  = np.array([
              [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)
    
    
    
    Aux_M6  = np.array([
              [-1 ],
              [ 0 ],
              [-1 ],
              [ 0 ],
              [-1 ],
              [ 0 ],
              [-1 ],
              [ 0 ]], dtype=np.float64)
    
    
    Aux_M71 = np.array([
              [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)
    
    
    Aux_M72 = np.array([
              [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ]], dtype=np.float64)
    
    
    
    Aux_M8  = np.array([
              [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ]], dtype=np.float64)
    
    
    Aux_Mb  = np.array([
              [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 , -1  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
              [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)


########################################################

    batch_size = self.params.batch_size
    pts_1_tile = self.pts_1_tile
    # Solve for H using DLT
    pred_h4p_tile = tf.expand_dims(self.pred_h4p, [2]) # BATCH_SIZE x 8 x 1
    # 4 points on the second image
    pred_pts_2_tile = tf.add(pred_h4p_tile, pts_1_tile)


    # Auxiliary tensors used to create Ax = b equation
    M1 = tf.constant(Aux_M1, tf.float32)
    M1_tensor = tf.expand_dims(M1, [0])
    M1_tile = tf.tile(M1_tensor,[batch_size,1,1])

    M2 = tf.constant(Aux_M2, tf.float32)
    M2_tensor = tf.expand_dims(M2, [0])
    M2_tile = tf.tile(M2_tensor,[batch_size,1,1])

    M3 = tf.constant(Aux_M3, tf.float32)
    M3_tensor = tf.expand_dims(M3, [0])
    M3_tile = tf.tile(M3_tensor,[batch_size,1,1])

    M4 = tf.constant(Aux_M4, tf.float32)
    M4_tensor = tf.expand_dims(M4, [0])
    M4_tile = tf.tile(M4_tensor,[batch_size,1,1])

    M5 = tf.constant(Aux_M5, tf.float32)
    M5_tensor = tf.expand_dims(M5, [0])
    M5_tile = tf.tile(M5_tensor,[batch_size,1,1])

    M6 = tf.constant(Aux_M6, tf.float32)
    M6_tensor = tf.expand_dims(M6, [0])
    M6_tile = tf.tile(M6_tensor,[batch_size,1,1])


    M71 = tf.constant(Aux_M71, tf.float32)
    M71_tensor = tf.expand_dims(M71, [0])
    M71_tile = tf.tile(M71_tensor,[batch_size,1,1])

    M72 = tf.constant(Aux_M72, tf.float32)
    M72_tensor = tf.expand_dims(M72, [0])
    M72_tile = tf.tile(M72_tensor,[batch_size,1,1])

    M8 = tf.constant(Aux_M8, tf.float32)
    M8_tensor = tf.expand_dims(M8, [0])
    M8_tile = tf.tile(M8_tensor,[batch_size,1,1])

    Mb = tf.constant(Aux_Mb, tf.float32)
    Mb_tensor = tf.expand_dims(Mb, [0])
    Mb_tile = tf.tile(Mb_tensor,[batch_size,1,1])

    # Form the equations Ax = b to compute H
    # Form A matrix
    A1 = tf.matmul(M1_tile, pts_1_tile) # Column 1
    A2 = tf.matmul(M2_tile, pts_1_tile) # Column 2
    A3 = M3_tile                   # Column 3
    A4 = tf.matmul(M4_tile, pts_1_tile) # Column 4
    A5 = tf.matmul(M5_tile, pts_1_tile) # Column 5
    A6 = M6_tile                   # Column 6
    A7 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M72_tile, pts_1_tile)# Column 7
    A8 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M8_tile, pts_1_tile)# Column 8

    A_mat = tf.transpose(tf.stack([tf.reshape(A1,[-1,8]),tf.reshape(A2,[-1,8]),\
                                   tf.reshape(A3,[-1,8]),tf.reshape(A4,[-1,8]),\
                                   tf.reshape(A5,[-1,8]),tf.reshape(A6,[-1,8]),\
         tf.reshape(A7,[-1,8]),tf.reshape(A8,[-1,8])],axis=1), perm=[0,2,1]) # BATCH_SIZE x 8 (A_i) x 8
    print('--Shape of A_mat:', A_mat.get_shape().as_list())
    # Form b matrix
    b_mat = tf.matmul(Mb_tile, pred_pts_2_tile)
    print('--shape of b:', b_mat.get_shape().as_list())

    # Solve the Ax = b
    H_8el = tf.matrix_solve(A_mat , b_mat)  # BATCH_SIZE x 8.
    print('--shape of H_8el', H_8el)


    # Add ones to the last cols to reconstruct H for computing reprojection error
    h_ones = tf.ones([batch_size, 1, 1])
    H_9el = tf.concat([H_8el,h_ones],1)
    H_flat = tf.reshape(H_9el, [-1,9])
    self.H_mat = tf.reshape(H_flat,[-1,3,3])   # BATCH_SIZE x 3 x 3

def transform(self):
    # Transform H_mat since we scale image indices in transformer
    H_mat = tf.matmul(tf.matmul(self.M_tile_inv, self.H_mat), self.M_tile)
    # Transform image 1 (large image) to image 2
    out_size = (self.params.img_h, self.params.img_w)
    warped_images, _ = transformer(self.I, H_mat, out_size)
    # TODO: warp image 2 to image 1
    
    
    # Extract the warped patch from warped_images by flatting the whole batch before using indices
    # Note that input I  is  3 channels so we reduce to gray
    warped_gray_images = tf.reduce_mean(warped_images, 3)
    warped_images_flat = tf.reshape(warped_gray_images, [-1])
    patch_indices_flat = tf.reshape(self.patch_indices, [-1])
    pixel_indices =  patch_indices_flat + self.batch_indices_tensor
    pred_I2_flat = tf.gather(warped_images_flat, pixel_indices)    
    self.pred_I2 = tf.reshape(pred_I2_flat, [self.params.batch_size, self.params.patch_size, self.params.patch_size, 1])