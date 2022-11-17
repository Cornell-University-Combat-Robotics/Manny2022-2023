import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from skimage import transform
from skimage.io import imread, imshow
import corner_detection
from PIL import Image

def warp(file_path):
    # code taken from https://towardsdatascience.com/image-processing-with-python-applying-homography-for-image-warping-84cd87d2108f

    sign = imread(file_path)
    height, width, depth = sign.shape
    print("width is ", width)
    print("height is", height)


    points_of_interest = np.array(corner_detection.get_corners(file_path))

    print("points of interest are", points_of_interest)
    if len(points_of_interest) != 4:
        raise Exception("Could not detect four corners in image. Instead detected " + str(len(points_of_interest)) + " corners" )
    #print("points of interest are", points_of_interest)

    # moves the image 50 pixels to the right (used for testing purposes)
    # corner_detection.get_corners('FourCorners2.jpg') is of type list
    # corners = corner_detection.get_corners(file_path)
    # for i in range(len(corners)):
    #     corners[i][0] += 50
    

    # projection = np.array(corners)
    ratio = 1.2 #ratio of height to width

    projection = np.array([[(width/2) - (2.5*height/12), height/4], [(width/2) + (2.5*height/12), height/4], [(width/2) - (2.5*height/12), 3*height/4], [(width/2) + (2.5*height/12), 3*height/4]]).astype(int)
    #projection = np.array([[height/4, (width/2) - (2.5*height/12)], [height/4,(width/2) + (2.5*height/12)], [3*height/4, (width/2) - (2.5*height/12)], [3*height/4,(width/2) + (2.5*height/12)]]).astype(int)
    print("projection is", projection)
    tform = transform.estimate_transform('projective', points_of_interest, projection)
    tf_img_warp = transform.warp(sign, tform.inverse, mode = 'symmetric')

    # uncomment the following to show the before/after image of the warped function
    # plt.figure(num=None, figsize=(8, 6), dpi=80)
    # fig, ax = plt.subplots(1,2, figsize=(15, 10), dpi = 80)
    # ax[0].set_title(f'Original', fontsize = 15)
    # ax[0].imshow(sign)
    # ax[0].set_axis_off();
    # ax[1].set_title(f'Transformed', fontsize = 15)
    # ax[1].imshow(tf_img_warp)
    # ax[1].set_axis_off();
    # plt.show()

    # both tf_img_warp and projection are of type numpy.ndarray
    return [tf_img_warp, projection]

# projection points are listed in "z" order
    # 1----------2
    # |          |
    # 3----------4
def crop(tf_img_warp, proj):
    minX = proj[0][0]
    maxX = proj[1][0]
    minY = proj[0][1] # remember that (0,0) is in the upper left corner
    maxY = proj[2][1]

    # minX = proj[0][0]
    # maxX = proj[2][0]
    # minY = proj[0][1] # remember that (0,0) is in the upper left corner
    # maxY = proj[1][1]
    print("minX is", minX)
    print("maxX is", maxX)
    print("minY is", minY)
    print("maxY is", maxY)
    return tf_img_warp[minY:maxY, minX:maxX]

# imshow('FourCorners2.jpg') # uncomment this to show the original image

arr = warp('Sebastian3.jpg')
# arr[0] is the warped image
# arr[1] are the projection points (corners of the warped image arranged in z order)
img = crop(arr[0], arr[1])
imshow(img)
plt.show()