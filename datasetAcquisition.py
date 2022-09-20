import numpy
import os
import sys
import csv
import scipy.io
import pickle
import time
import datetime
import glob
import numpy
import cv2
from sklearn.cluster import KMeans

###

def BGR2HSV(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) * 1.
    image_hsv[:,:,0] /= 180.
    image_hsv[:,:,1] /= 255.
    image_hsv[:,:,2] /= 255.
    return image_hsv

def HSV2BGR(image):
    image_hsv = image.copy()
    image_hsv[:,:,0] *= 180.
    image_hsv[:,:,1] *= 255.
    image_hsv[:,:,2] *= 255.
    image_rgb = cv2.cvtColor(image_hsv.astype(numpy.uint8), cv2.COLOR_HSV2BGR) * 1.
    return image_rgb

def blur(image_name):
    image = cv2.imread("images/"+image_name+".png")
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred_v = cv2.bilateralFilter(image_hsv[:,:,2], 101, sigmaColor=50, sigmaSpace=51)
    blurred_v = cv2.ximgproc.guidedFilter(image.max(axis=2), blurred_v, 40, 0.01)
    return blurred_v

def readData(image_name, woilluminationmap, woactivelearning):
    input_image_original = BGR2HSV(cv2.imread("images/"+image_name+".png"))
    illumination_original = cv2.imread("LIME/illumination/"+image_name+".png", cv2.IMREAD_GRAYSCALE) / 255.
    illumination_original[illumination_original>=0.5] = 0.5
    illumination_original = cv2.ximgproc.guidedFilter((input_image_original[:,:,2]*255).astype(numpy.uint8), (illumination_original * 255).astype(numpy.uint8), 10, 100) / 255.
    height, width = input_image_original.shape[:2]
    blur_image_original = blur(image_name)

    x_original = \
        numpy.asmatrix(numpy.zeros((height*width, 6)))
    for i in range(height):
        for j in range(width):
            x_original[i*width+j, 0] = (i - height/2.) * 1. / max(height, width) / 8. / 10
            x_original[i*width+j, 1] = (j - width/2.) * 1. / max(height, width) / 8. / 10
            x_original[i*width+j, 2] = (input_image_original[i, j, 0] - 0.5).reshape(1, 1) * 0
            x_original[i*width+j, 3] = (input_image_original[i, j, 1] - 0.5).reshape(1, 1) * 0
            x_original[i*width+j, 4] = (input_image_original[i, j, 2] - 0.5).reshape(1, 1) * 0.75 * 0
            if not woilluminationmap:
                x_original[i*width+j, 5] = (illumination_original[i, j] - 0.25).reshape(1, 1) * 0.75
            else:
                x_original[i*width+j, 5] = (illumination_original[i, j] - 0.25).reshape(1, 1) * 0

    if not woactivelearning:
        kmeans_cls = KMeans(n_clusters=256, n_init=1, max_iter=10)
        kmeans_result = kmeans_cls.fit(x_original)
        kmeans_u, kmeans_counts_ = numpy.unique(kmeans_result.labels_, return_counts=True)
        kmeans_counts = numpy.zeros(256)
        kmeans_counts[kmeans_u] = kmeans_counts_
        kmeans_centers = kmeans_result.cluster_centers_
        kmeans_result_labels_ = kmeans_result.labels_
    else:
        kmeans_centers = numpy.random.permutation(x_original.copy())
        kmeans_counts = numpy.ones(x_original.shape[0])
        kmeans_result_labels_ = numpy.arange(x_original.shape[0])

    return x_original, numpy.asmatrix(kmeans_centers), numpy.asmatrix(kmeans_counts[:,None]), \
                input_image_original, kmeans_result_labels_, image_name, blur_image_original


def readDataLPF(image_name):
    input_image_original = BGR2HSV(cv2.imread("images/"+image_name+".png"))
    blur_image_original = blur(image_name)

    return input_image_original, image_name, blur_image_original


def readDataSLS(image_name):
    input_image_original = BGR2HSV(cv2.imread(glob.glob("images/"+image_name+".*")[0]))

    return input_image_original, image_name
