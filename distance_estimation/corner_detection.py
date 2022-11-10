import math
from image_distance import headshot_ratio
import numpy as np
from PIL import Image
import sys
from operator import itemgetter
from image_distance import *
import time

start_time = time.time()

#from IPython.display import display
#from ipywidgets import interact
np.set_printoptions(threshold=sys.maxsize)


img = np.array(Image.open('FourCorners2.jpg'))
print(img.shape)
height, width = img.shape[0], img.shape[1]

r=0
g=1
b=2

r_query = 135
g_query = 135
b_query = 70
#print("hello")
Y,X = np.where((img[:,:,r] >= r_query) & (img[:,:,g] >= g_query) & (img[:,:,b] <= b_query))
y_points = np.column_stack((X,Y))
#print(y_points)
#sy_points = sorted(y_points, key=lambda tup: tup[0] * 5 + tup[1]*10)

min_x = min(y_points, key = lambda x: x[0])[0]
min_y = min(y_points, key = lambda x: x[1])[1]
max_x = max(y_points, key = lambda x: x[0])[0]
max_y = max(y_points, key = lambda x: x[1])[1]

x_thresh = (max_x - min_x)/2
y_thresh = (max_y - min_y)/2

#y_points.sort(key=(itemgetter(0) * 5) + (itemgetter(1) * 10))
#print(sy_points[0])
#for i in range(len(sy_points)):
#    print(sy_points[i])

#get threshhold by finding minimum and maximum x and y
#print(x_thresh)
#print(y_thresh)
fps= []
for p in y_points:
    #if x or y differ substantially from every other point's x and y
    Diff=True
    for p2 in fps:
        if (abs(p2[0] - p[0]) < x_thresh and abs(p2[1] - p[1]) < y_thresh):
            Diff=False
            break
    if Diff:
        #print(p)
        fps.append(p)

fps = sorted(fps, key=lambda tup: tup[0] * 5 + tup[1]*10)
#print(fps)


# leny= len(sy_points)
# c1 = sy_points[0]
# c2 = sy_points[int(leny*(1/8))]
# c3 = sy_points[int(leny*(5/8))]
# c4 = sy_points[leny-1]

print(headshot_ratio(fps[0][0],fps[0][1],fps[1][0],fps[1][1],fps[3][0],fps[3][1], fps[2][0],fps[2][1], width, height))

for p in fps:
    img[p[1], p[0]] = np.array([255,0,0])

img = Image.fromarray(img, 'RGB')
img.show()

print("--- %s seconds ---" % (time.time() - start_time))




#print(yellow_points)
#print(yellow_points.shape())
#print(yellow_points[1][5])
# print(yellow_points)
# leny = len(yellow_points[0])
# c1 = (yellow_points[0][0], yellow_points[1][0])
# c2 = (yellow_points[0][int(leny*(3/8))], yellow_points[1][int(leny*(3/8))])
# c3 = (yellow_points[0][int(leny*(5/8))], yellow_points[1][int(leny*(5/8))])
# c4 = (yellow_points[0][leny-1], yellow_points[1][leny-1])


# print(yellow_points.index)
# print(c1)
# print(c2)
# print(c3)
# print(c4)

#print(yellow_points[0][0])
#print(yellow_points[1][0])
#print(yellow_points[len(yellow_points)//2])