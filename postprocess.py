#!/usr/bin/env python3

"""
Script Name: postprocess.py
Author: Stefan Herdy
Date: 25.07.2023
Description: 
Description: 
Script to postprocess images by cropping spores and adding white padding around them. 
The processed images are saved in a specified output directory.

Usage:
- Make sure that you have run the sam.py script to get the pre-selected images.
- Place the images to be processed in the input directory specified by 'root_folder'.
- Specify the output directory where the processed images will be saved using 'destination_folder'.
- Run the script to process the images and save the results.
"""

import os
import cv2
import numpy as np
import glob
import argparse

def add_white_padding(image, padding_size=30):
    # Get the current image size
    height, width, channels = image.shape

    # Calculate the new size after adding padding
    new_height = height + 2 * padding_size
    new_width = width + 2 * padding_size

    # Create a white background with the new size
    padded_image = 255 * np.ones((new_height, new_width, channels), dtype=np.uint8)

    # Calculate the positions to place the original image
    x_position = padding_size
    y_position = padding_size

    # Place the original image in the center of the white background
    padded_image[y_position:y_position+height, x_position:x_position+width] = image

    return padded_image

def keep_biggest_connected_mask(threshold_array):
    # Find connected components in the threshold array
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(threshold_array.astype(np.uint8))
    if num_labels > 1:
        # Find the index of the largest connected component (excluding the background)
        largest_component_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

        # Create a new array with only the largest connected component set to True
        biggest_mask = np.where(labels == largest_component_idx, 1, 0).astype(np.bool)
    else:
        biggest_mask = threshold_array

    return biggest_mask
    

def crop_spores_in_directory(input_dir, output_dir, min_spore_area=500):
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        for file in os.listdir(subdir_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, subdir, file)
                output_subdir_path = os.path.join(output_dir, subdir)
                ex = glob.glob(os.path.join(output_subdir_path, file))
                if len(ex) > 0:
                    print('Image already processed!')
                if len(ex) == 0:
                    
                    if not os.path.exists(output_subdir_path):
                        os.makedirs(output_subdir_path)

                    img = cv2.imread(input_path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, (threshold) = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)   
                    threshold = keep_biggest_connected_mask(threshold)
                    
                    segmentation = np.where(threshold == True)
                    if len(segmentation[0]) > 0:
                        x_min = int(np.min(segmentation[1]))
                        x_max = int(np.max(segmentation[1]))
                        y_min = int(np.min(segmentation[0]))
                        y_max = int(np.max(segmentation[0]))

                        cropped = img[y_min:y_max, x_min:x_max]
                        cropped = add_white_padding(cropped)
                        output_path = os.path.join(output_subdir_path, file)
                        cv2.imwrite(output_path, cropped)
                        print('Saving image to: ' + file)
                    else:
                        print('Image contains no spores!')
                    

if __name__ == "__main__":
    # Data path
    # Select your own data path! To try the script there are some images stored under "./imgs/set_1/"
    root_folder = "path/to/your/data/"
    # Path to store the segmented images
    destination_folder = "path/to/your/destination/folder/"

    crop_spores_in_directory(root_folder, destination_folder)
