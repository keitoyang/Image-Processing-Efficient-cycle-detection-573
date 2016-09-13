# Author: Kun Yang and Hengtong Zhang

import numpy as np
import cv2
import time
import math
from scipy import signal
from scipy.spatial import distance

img = cv2.imread('HoughCircles.jpg', 0)
img_color = cv2.imread('HoughCircles.jpg')

start = time.time()

# (a)Denoise the image using a Gaussian blur filter
Gaussian = np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]])

Denoised_img = signal.convolve2d(img, Gaussian, boundary='symm', mode='same')
Denoised_img /= 16
# cv2.imwrite('Denoised.jpg', Denoised_img)
print('(a) Denoise the image using a Gaussian blur filter......Done')

# (b & c)Apply an edge detector. Threshold the image from (b) into a binary image
Denoised_img = np.uint8(Denoised_img)
Edge_img = cv2.Canny(Denoised_img, 30, 100)
# cv2.imwrite('Edge.jpg', Edge_img)
# cv2.imshow('image', Edge_img)
# cv2.waitKey(0)
print('(b & c) Apply an edge detector......Done')

# (d)Pick a suitable range of radii values of circles
#   We choose 10~60 for raddi values, the smallest 
#   circle in the image has radius of 19, the largest
#   circle in the image has radius of 51. So 10~60 
#   can cover all possible circles, and the amount
#   of computation can be reduced if we have smaller range.

LOW = 10
HIGH = 60

X_list = []
Y_list = []
for a in xrange(LOW, HIGH + 1):
    xl = []
    yl = []
    for x in xrange(-a, a + 1):
        for y in xrange(-a, a + 1):
            if int(round(math.sqrt(x ** 2 + y ** 2))) == a:
                xl.append(x)
                yl.append(y)
    X_list.append(xl)
    Y_list.append(yl)

# (e)Apply Hough transform to detect circle
#   Step One: construct the 3D accumulator
#   Step Two: For every edge pixel, modify the corresponding accumulator position
#   Step Three: Scan the accumlator, find the highest values

accumulator = [[[0 for x in xrange(img.shape[0])] for y in xrange(img.shape[1])] for r in xrange(HIGH - LOW + 1)]

for x in xrange(img.shape[0]):
    for y in xrange(img.shape[1]):
        if Edge_img[x][y] == 255:
            for r in xrange(HIGH - LOW + 1):
                for (a, b) in zip(X_list[r], Y_list[r]):
                    if 0 < (x + a) < img.shape[0]:  # Optimize by: Hengtong
                        if img.shape[1] > (y + b) > 0:
                            accumulator[r][y + b][x + a] += 1

X = []
Y = []
R = []
S = []
for r in xrange(HIGH - LOW + 1):
    for y in xrange(img.shape[1]):
        for x in xrange(img.shape[0]):
            if accumulator[r][y][x] > 90:
                X.append(x)
                Y.append(y)
                R.append(r)
                S.append(accumulator[r][y][x])

for i in xrange(len(X)):
    for j in xrange(i + 1, len(X)):
        if distance.euclidean((X[i], Y[i]), (X[j], Y[j])) < 20:
            if S[i] < S[j]:
                R[i] = 0
            else:
                R[j] = 0
print('(d & e) Apply Hough transform to detect circle......Done')

# (f)Display the circles detected over original image
for i in xrange(len(X)):
    if R[i] != 0:
        for (a, b) in zip(X_list[R[i]], Y_list[R[i]]):
            if 0 < (X[i] + a) < img.shape[0]:
                if img.shape[1] > (Y[i] + b) > 0:   # Optimize by: Hengtong
                    img_color[X[i] + a][Y[i] + b] = [0, 0, 255]  # blue circles

cv2.imwrite('Result.jpg', img_color)
cv2.imshow('image', img_color)
cv2.waitKey(0)
print('(f) Display the circles detected over original image...... time:' + str(time.time() - start))





