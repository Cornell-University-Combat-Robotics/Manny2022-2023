import math
from image_distance import headshot_ratio
import numpy as np
from PIL import Image
import sys
from operator import itemgetter
from image_distance import *
import time

np.set_printoptions(threshold=sys.maxsize)

def get_corners(img_path):
    img = np.array(Image.open(img_path))
    height, width = img.shape[0], img.shape[1]

    r=0
    g=1
    b=2

    r_query = 140
    g_query = 150
    b_query = 140

    Y,X = np.where((img[:,:,r] <= r_query) & (img[:,:,g] >= g_query) & (img[:,:,b] <= b_query))
    y_points = np.column_stack((X,Y))

    min_x = min(y_points, key = lambda x: x[0])[0]
    min_y = min(y_points, key = lambda x: x[1])[1]
    max_x = max(y_points, key = lambda x: x[0])[0]
    max_y = max(y_points, key = lambda x: x[1])[1]

    x_thresh = (max_x - min_x)/2
    y_thresh = (max_y - min_y)/2

    # fps = final points
    fps= []
    for p in y_points:
        # if x or y differ substantially from every other point's x and y
        Diff=True
        for p2 in fps:
            if (abs(p2[0] - p[0]) < x_thresh and abs(p2[1] - p[1]) < y_thresh):
                Diff=False
                break
        if Diff:
            #print(p)
            fps.append(p)

    # sorted in a z manner
    # 1----------2
    # |          |
    # 3----------4
    fps = sorted(fps, key=lambda tup: tup[0] * 5 + tup[1]*10)

    for p in fps:
        img[p[1], p[0]] = np.array([255,0,0])

    # img = Image.fromarray(img, 'RGB')
    # img.show()

    # (x coordinate, y coordinate)
    return fps

#print(get_corners('Blaze2.jpg'))