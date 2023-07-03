#!/usr/bin/env python3

"""
Script Name: utols.py
Author: Stefan Herdy
Date: 15.06.2023
Description: 
This is a code implementation of Facebooks "Segment Anything Model"
A sample script how to utilize the Segment Anything Model to automatically segment microscopy images of spores

Usage: 
-  Download the desired Segment Anything Models from the original GitHub page:
   https://github.com/facebookresearch/segment-anything
-  This script contains helper functions for sam.py
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

def remove_scale(image, show = False):
    #image = cv2.imread(image_path)
    #im_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = 15
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.dilate(mask[1], kernel, iterations=7)
    image[mask == 255] = [255,255,255]
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    numpy_horizontal = np.hstack((image, mask))
    if show == True:
        cv2.imshow('Image w/o Scale',numpy_horizontal)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# just for tests, remove_scale('C:/Users/stefa/Desktop/repos/use-segment-anything-model-to-autosegment-microscope-images/riccia_imgs_selected/Riccia beyrichiana/ABry_531_Riccia_beyrichiana_proximal_rem31.png')