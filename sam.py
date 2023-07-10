#!/usr/bin/env python3

"""
Script Name: sam.py
Author: Stefan Herdy
Date: 15.06.2023
Description: 
This is a code implementation of Facebooks "Segment Anything Model"
A sample script how to utilize the Segment Anything Model to automatically segment microscopy images of spores

Usage: 
-  First, download the desired Segment Anything Models from the original GitHub page:
   https://github.com/facebookresearch/segment-anything
-  Copy the models to your model folder
-  Set your data path, your destination path and your model path and run the script
"""


from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os 
import glob
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from skimage.transform import resize
from utils import remove_scale

# Data path
root_folder = "C:/Users/stefa/Desktop/repos/use-segment-anything-model-to-autosegment-microscope-images/riccia_imgs/"
# Path to store the segmented images
destination_folder = "C:/Users/stefa/Desktop/repos/use-segment-anything-model-to-autosegment-microscope-images/riccia_imgs_selected"

# Model path
ckpt_vit_b = "C:/Users/stefa/Desktop/repos/use-segment-anything-model-to-autosegment-microscope-images/sam_models/sam_vit_b_01ec64.pth"
ckpt_vit_h = "C:/Users/stefa/Desktop/repos/use-segment-anything-model-to-autosegment-microscope-images/sam_models/sam_vit_h_4b8939.pth"

# Init model
sam = sam_model_registry["vit_b"](checkpoint=ckpt_vit_b)
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

isresize = True
resize_factor = 0.5

# Iterate through folder
for subdir in os.listdir(root_folder):
    subdir_path = os.path.join(root_folder, subdir)
    if os.path.isdir(subdir_path) and "Riccia" in subdir:
        for image_name in os.listdir(subdir_path):
            #image = Image.open(subdir_path + '/' + image_name)
            image = cv2.imread(subdir_path + '/' + image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if isresize == True:
                image_orig = image
                width = int(image.shape[1] * resize_factor)
                height = int(image.shape[0] * resize_factor)
                dim = (width, height)
                # resize image
                image = cv2.resize(image, dim, interpolation = cv2.INTER_NEAREST)
            if not os.path.isdir(destination_folder + '/' + subdir):
                os.makedirs(destination_folder + '/' + subdir)
            #image.save(destination_folder + '/' + subdir + '/'  + image_name)
            new_name = destination_folder + '/' + subdir + '/'  + os.path.splitext(image_name)[0] + '_' + '*' + '.png'
            print('Name: ' + new_name)
            ex = glob.glob(new_name)
            if len(ex) > 0:
                print('Image already processed!')
            if len(ex) == 0:
                # Generate the segmentation masks
                masks = mask_generator.generate(image)
                print('Masks detected:' + str(len(masks)))
                for i in range(len(masks)):
                    image_new = image_orig.copy()
                    bool_mask = masks[i]['segmentation']
                    if isresize == True:
                        bool_mask = resize(bool_mask, (image_new.shape[0], image_new.shape[1]))
                    #labeled_image, count = skimage.measure.label(bool_mask, return_num=True)
                    #object_features = skimage.measure.regionprops(labeled_image)
                    #object_areas = [objf["area"] for objf in object_features]
                    #for object_id, objf in enumerate(object_features, start=1):
                    #    if objf["area"] < max(object_areas):
                    #        labeled_image[labeled_image == objf["label"]] = False
                    #    if objf["area"] == max(object_areas):
                    #        labeled_image[labeled_image == objf["label"]] = True

                    # White background
                    image_new[bool_mask == False] = [255,255,255]
                    image_new = cv2.cvtColor(image_new, cv2.COLOR_RGB2BGR)

                    area = masks[i]['area']
                    area_thresh = image.shape[0]/10*image.shape[1]/10
                    # Save image if area is bigger than a specified threshold 
                    if area > area_thresh:
                        # We assume, that there are no spores etc. at the corners of the images.
                        # If the mask is at the corner of the images we know that the mask is the background mask and do not save it.
                        # Use coordinates [10,10] instead of [0,0], because the model sometimes has problems with the image edges and labels the first few pixel rows incorrect
                        if bool_mask[10,10] == False:
                            cv2.imwrite(destination_folder + '/' + subdir + '/'  + os.path.splitext(image_name)[0] + '_' + str(i) + '.png', image_new)
            # Sometimes the models label everything but the object of interest. 
            # So if still no mask is detected, iteration is done over the "negative" masks.
            ex = glob.glob(new_name)
            if len(ex) == 0:
                print('Computing reverse mask: ')
                image_new = image_orig.copy()
                remaining_mask = np.full((image_new.shape[0], image_new.shape[1]), False)
                #if isresize == True:
                #        bool_mask = resize(bool_mask, (image_new.shape[0], image_new.shape[1]))
                for i in range(len(masks)):
                    if isresize == True:
                        mask = resize(masks[i]['segmentation'], (image_new.shape[0], image_new.shape[1]))
                    else:
                        mask = masks[i]['segmentation']

                    mean_val = cv2.mean(image_new, mask.astype(np.uint8))
                    mean_val = np.mean(mean_val[0:2])
                    if mean_val > 200: # or mean_val < 40:
                        remaining_mask = remaining_mask + mask
                # labeled_image, count = skimage.measure.label(remaining_mask, return_num=True)
                # object_features = skimage.measure.regionprops(labeled_image)
                # object_areas = [objf["area"] for objf in object_features]
                # for object_id, objf in enumerate(object_features, start=1):
                #     if objf["area"] < max(object_areas):
                #         labeled_image[labeled_image == objf["label"]] = False
                #     if objf["area"] == max(object_areas):
                #         labeled_image[labeled_image == objf["label"]] = True
                # White background
                if isresize == True:
                    remaining_mask = resize(remaining_mask, (image_new.shape[0], image_new.shape[1]))
                image_new[remaining_mask == True] = [255,255,255]
                image_new = cv2.cvtColor(image_new, cv2.COLOR_RGB2BGR)
                area_all = 0
                for j in range(len(masks)):
                    area_all = area_all + masks[j]['area']
                area = image.shape[0]*image.shape[1] - area_all
                area_thresh = image.shape[0]/15*image.shape[1]/15
                # Save image if area is bigger than a specified threshold 
                if area > area_thresh:
                    # We assume, that there are no spores etc. at the corners of the images.
                    # If the mask is at the corner of the images we know that the mask is the background mask and do not save it.
                    # Use coordinates [10,10] instead of [0,0], because the model sometimes has problems with the image edges and labels the first few pixel rows incorrect
                    if remaining_mask[10,10] == True:
                        # image_new = remove_scale(image_new)
                        cv2.imwrite(destination_folder + '/' + subdir + '/'  + os.path.splitext(image_name)[0] + '_rem' + str(i) + '.png', image_new)
            del image
            if len(ex) == 0:
                del image_new, image_orig