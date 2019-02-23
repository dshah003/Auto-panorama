
import os
import cv2
import glob
import cPickle as pickle
import random
import numpy as np
#import scipy.misc
#from skimage import color
#from skimage import io
import matplotlib.pyplot as plt


perturbation_limit = 30

image_dir = '../Data/Stacked/'


if not os.path.exists(image_dir):
    os.makedirs(image_dir)




def create_patch(image):
    h,w = image.shape
    
    if h > 250 and w >250:
        Ca = [int(h/2)-100, int(h/2)+100, int(w/2)-100, int(w/2)+100]
        P_a = image[Ca[0]:Ca[1], Ca[2]:Ca[3]]
    else:
        Ca = [int(h/2)-50, int(h/2)+50, int(w/2)-50, int(w/2)+50]
        P_a = image[Ca[0]:Ca[1], Ca[2]:Ca[3]]
#     print (Ca)
    #Arrange Coordinates into array
    Corner = np.array([
        [Ca[2], Ca[0]],
        [Ca[3], Ca[0]],
        [Ca[3], Ca[1]],
        [Ca[2], Ca[1]]], dtype = "int32")
    # print(Corner)
    return P_a, Corner





def purturb(CornerA,perturbation_limit):
    
    CornerB = np.zeros((4,2),dtype=np.int)
    for i in range(4):
        for j in range(2):
            CornerB[i][j] = CornerA[i][j]+random.randint(-perturbation_limit,perturbation_limit)
            
    return CornerB
  





def pertb_size(CornerB):
    xs = []
    ys = []
    for i in range(4):
        xs.append(CornerB[i][0])
        ys.append(CornerB[i][1])
    
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    h = int((y_max-y_min))
    w = int((x_max-x_min))
    
    return h,w


# importing images from folder

img_dir = "../Data/Train" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
all_stacked = []
all_H4pt = []
i = 0

print("Generating Images for Network\n")
for f1 in files:
    Img1 = cv2.imread(f1,cv2.IMREAD_GRAYSCALE)
#     data.append(img)
#     # Plotting rectangle on the image


#     Img1 = data[0]
    h,w = Img1.shape
    Img1_copy = Img1.copy()    
    patched_image,CornerA = create_patch(Img1)
    CornerB = purturb(CornerA,perturbation_limit)
# cv2.polylines(Img1_copy,[CornerB],True,(0,0,255))
# plt.imshow(Img1_copy, cmap= 'gray')

#     cv2.polylines(Img1_copy,[CornerA],True,(0,0,255))
#     plt.imshow(Img1_copy, cmap= 'gray')

    CornerA = CornerA.astype(np.float32)
    CornerB = CornerB.astype(np.float32)

    HAB = cv2.getPerspectiveTransform(CornerA, CornerB)
    HBA = np.linalg.inv(HAB)

    p_h,p_w = pertb_size(CornerB)
    dst = cv2.warpPerspective(Img1,HBA,(w,h))

    P_b,CornerR = create_patch(dst)

    # plt.subplot(1,2,1)
    # plt.imshow(patched_image, cmap= 'gray')

    # plt.subplot(1,2,2)
    # plt.imshow(P_b, cmap= 'gray')

    stacked = np.stack([patched_image,P_b],axis=2)
    # stacked.shape
    
    stacked = cv2.resize(stacked,(128,128))

    H4pt = CornerB-CornerA
    name = str(i)+'.jpg'
    path = image_dir+name
    
#     stacked = stacked.astype('uint8')
#     scipy.misc.imsave(path, stacked)
    
#     cv2.imwrite(path,stacked)
    all_stacked.append(stacked)
    all_H4pt.append(H4pt)
    
    i += 1
        
pickle.dump( all_H4pt, open( image_dir +"all_H4pt.p", "wb" ) )
pickle.dump( all_stacked, open( image_dir +"all_stacked_images.p", "wb" ) )

print("Done!!")

# color = pickle.load( open( image_dir + "all_H4pt.p", "rb" ) )

# plt.imshow(dst,cmap = 'gray')
