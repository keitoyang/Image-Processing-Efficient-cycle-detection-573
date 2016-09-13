# Author: Kun Yang and Hengtong Zhang

import numpy as np
import cv2
import time
import math
from scipy import signal
from scipy.spatial import distance

# Bonus section
circles = []
V = []
f = 0
Tf = 30000
Tmin = 60
Ta = 2
Td = 4
Tr = 0.6

LOW = 10
HIGH = 60

img = cv2.imread('HoughCircles.jpg', 0)
img_color = cv2.imread('HoughCircles.jpg')

start = time.time()


def get_center(p1, p2, p3):
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = p3
    a = [[x1 ** 2 + y1 ** 2, y1, 1],
         [x2 ** 2 + y2 ** 2, y2, 1],
         [x3 ** 2 + y3 ** 2, y3, 1]]
    b = [[x1, x1 ** 2 + y1 ** 2, 1],
         [x2, x2 ** 2 + y2 ** 2, 1],
         [x3, x3 ** 2 + y3 ** 2, 1]]
    c = [[x1, y1, 1],
         [x2, y2, 1],
         [x3, y3, 1]]
    a = np.linalg.det(a)
    b = np.linalg.det(b)
    c = np.linalg.det(c)
    #     |x1^2+y1^2  y1  1|        |x1  x1^2+y1^2  1|
    #     |x2^2+y2^2  y2  1|        |x2  x2^2+y2^2  1|
    #     |x3^2+y3^2  y3  1|        |x3  x3^2+y3^2  1|
    # h = ------------------,   k = ------------------
    #         |x1  y1  1|               |x1  y1  1|
    #       2*|x2  y2  1|             2*|x2  y2  1|
    #         |x3  y3  1|               |x3  y3  1|
    if c != 0:
        h = a / (2 * c)
        k = b / (2 * c)
    else:
        h = k = None
    vec = (h, k)
    return vec


if __name__ == '__main__':
    # (a)Denoise the image using a Gaussian blur filter
    Gaussian = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]])

    Denoised_img = signal.convolve2d(img, Gaussian, boundary='symm', mode='same')
    Denoised_img /= 16
    Denoised_img = np.uint8(Denoised_img)
    Edge_img = cv2.Canny(Denoised_img, 30, 200)

    for x in xrange(img.shape[0]):
        for y in xrange(img.shape[1]):
            if Edge_img[x][y] == 255:
                V.append((x, y))

    while True:
        if f == Tf or len(V) < Tmin:
            print("Algorithm finished!")
            break
        else:
            # Step 1 and 2
            rand_set = np.random.permutation(len(V)).tolist()[:4]
            recy = []
            for i in rand_set:  # we randomly pick four pixels vi , i = 1, 2, 3, 4, out of V.
                recy.append(V[i])

            for item in recy:
                V.remove(item)  # When vi has been chosen, set V = V - {vi}.

            # Step 3
            candidates = []
            possibles = [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 3, 1], [1, 2, 3, 0]]
            for item in possibles:
                dist1 = distance.euclidean(recy[item[0]], recy[item[1]])
                dist2 = distance.euclidean(recy[item[1]], recy[item[2]])
                dist3 = distance.euclidean(recy[item[2]], recy[item[0]])
                if dist1 > Ta and dist2 > Ta and dist3 > Ta:
                    (x1, y1) = recy[item[0]]
                    (x2, y2) = recy[item[1]]
                    (x3, y3) = recy[item[2]]
                    o = get_center((x1, y1), (x2, y2), (x3, y3))  # The o point of the circle
                    if o[0] is None or o[1] is None:
                        continue
                    r = distance.euclidean(recy[item[0]], o)
                    if math.fabs(distance.euclidean(recy[item[3]], o) - r) > Td:
                        candidates.append(item)  # Put it into candidate set.

            if len(candidates) > 0:
                # Step 4
                for item in candidates:
                    C = 0
                    pt_i = recy[item[0]]
                    pt_j = recy[item[1]]
                    pt_k = recy[item[2]]
                    o = get_center(pt_i, pt_j, pt_k)  # The o point of the circle
                    if o[0] is None or o[1] is None:
                        continue
                    r_ijk = distance.euclidean(pt_i, o)
                    pix_recy = []
                    for pix in V:
                        dl = math.fabs(distance.euclidean(pix, o) - r_ijk)
                        if dl <= Td:  # dl->ijk is not larger than the given distance threshold Td
                            C += 1
                            pix_recy.append(pix)
                    # if C >= r_ijk * 3 and HIGH > r_ijk > LOW:
                    if C >= r_ijk * 3:
                        # The possible circle Cijk has been detected as a true circle
                        for pix_r in pix_recy:
                            V.remove(pix_r)  # take vk out of V
                        tmp = [pt_i, pt_j, pt_k, o, r_ijk]
                        circles.append(tmp)
                        f = 0
                    else:
                        f += 1  # return these n_p edge pixels into V
                        break  # TODO: Return to step 2.
            else:
                f += 1
                for item in recy:
                    V.append(item)  # vi , i = 1, 2, 3, 4, back to V

    print('time:' + str(time.time() - start))
    # (f)Display the circles detected over original image
    for item in circles:
        x = int(item[3][0])
        y = int(item[3][1])
        cv2.circle(img_color, (y, x), int(item[4]), color=[0, 0, 255])
    cv2.imwrite('Result.jpg', img_color)
