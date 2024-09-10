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
import skimage
from utils import remove_scale
from datetime import datetime
import argparse

def segment_images(args, subdir_path, image_name, destination_folder):
    cnt = 0
    image = cv2.imread(os.path.join(subdir_path, image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_orig = image

    if image.shape[0] > args.size_thresh or image.shape[1] > args.size_thresh:
        args.isresize = True
    else:
        args.isresize = False
    
    if args.isresize == True:
        width = int(image.shape[1] * args.resize_factor)
        height = int(image.shape[0] * args.resize_factor)
        dim = (width, height)
        # resize image
        image = cv2.resize(image, dim, interpolation = cv2.INTER_NEAREST)

    if not os.path.isdir(os.path.join(destination_folder, subdir)):
        os.makedirs(os.path.join(destination_folder, subdir))

    # Check if image already processed
    all_masks_name = f'{destination_folder}/{subdir}/{os.path.splitext(image_name)[0]}_*.png'
    print(f'Name: {subdir}/{image_name}')
    processed_imgs = glob.glob(all_masks_name)
    if len(processed_imgs) > 0:
        print('Image already processed!')
    if len(processed_imgs) == 0:
        # Generate the segmentation masks
        masks = mask_generator.generate(image)
        print(f'Masks detected: {str(len(masks))}')
        for i in range(len(masks)):
            image_new = image_orig.copy()
            bool_mask = masks[i]['segmentation']
            if args.isresize == True:
                bool_mask = resize(bool_mask, (image_new.shape[0], image_new.shape[1]))
            
            # White background
            image_new[bool_mask == False] = [255,255,255]
            image_new = cv2.cvtColor(image_new, cv2.COLOR_RGB2BGR)

            area = masks[i]['area']
            area_thresh = image.shape[0]*image.shape[1]*args.area_thresh_ratio
            # Save image if area is bigger than a specified threshold 
            if area > area_thresh:
                # We assume, that there are no spores etc. at the corners of the images.
                # If the mask is at the corner of the images we know that the mask is the background mask and do not save it.
                # Use coordinates [10,10] instead of [0,0], because the model sometimes has problems with the image edges and labels the first few pixel rows incorrect
                if bool_mask[10,10] == False:
                    cv2.imwrite(f'{destination_folder}/{subdir}/{os.path.splitext(image_name)[0]}_{str(i)}.png', image_new)
                    cnt += 1
                    print(f'Number of analyzed Images: {str(cnt)}')

    # Sometimes the models label everything but the object of interest. 
    # So if still no mask is detected, iteration is done over the "negative" masks.
    processed_imgs = glob.glob(all_masks_name)
    if len(processed_imgs) == 0:
        print('Computing reverse mask: ')
        image_new = image_orig.copy()
        background_mask = np.full((image_new.shape[0], image_new.shape[1]), False)

        for i in range(len(masks)):
            if args.isresize == True:
                mask = resize(masks[i]['segmentation'], (image_new.shape[0], image_new.shape[1])).astype(np.int)
            else:
                mask = masks[i]['segmentation']

            mean_val = cv2.mean(image_new, mask.astype(np.uint8))
            mean_val = np.mean(mean_val[0:2])

            # We add the white (light) background to subtract it from the image later 
            if mean_val > 200:
                background_mask = background_mask + mask

        # White background
        if args.isresize == True:
            background_mask = resize(background_mask, (image_new.shape[0], image_new.shape[1])).astype(np.int)
        object_mask = ~background_mask

        # Just keep largest mask 
        labeled_image, count = skimage.measure.label(object_mask, return_num=True)
        object_features = skimage.measure.regionprops(labeled_image)
        object_areas = [objf["area"] for objf in object_features]
        for object_id, objf in enumerate(object_features, start=1):
            if objf["area"] < max(object_areas):
                labeled_image[labeled_image == objf["label"]] = False
            if objf["area"] == max(object_areas):
                labeled_image[labeled_image == objf["label"]] = True

        image_new[labeled_image == False] = [255,255,255]
        image_new = cv2.cvtColor(image_new, cv2.COLOR_RGB2BGR)
        area_all = 0
        for j in range(len(masks)):
            area_all = area_all + masks[j]['area']
        area = image.shape[0]*image.shape[1] - area_all
        area_thresh = image.shape[0]*image.shape[1]*args.area_thresh_ratio
        # Save image if area is bigger than a specified threshold 
        if area > area_thresh:
            # We assume, that there are no spores etc. at the corners of the images.
            # If the mask is at the corner of the images we know that the mask is the background mask and do not save it.
            # Use coordinates [10,10] instead of [0,0], because the model sometimes has problems with the image edges and labels the first few pixel rows incorrect
            if background_mask[10,10] == True:
                # image_new = remove_scale(image_new)
                cv2.imwrite(f'{destination_folder}/{subdir}/{os.path.splitext(image_name)[0]}_{str(i)}_revmask.png', image_new)
                cnt += 1
                print(f'Number of analyzed Images: {str(cnt)}')
    # Delete loaded images to release memory and prevent loop from slowing dowm
    del image
    if len(processed_imgs) == 0:
        del image_new, image_orig



if __name__ == "__main__":
    parser = argparse.ArgumentParser("AutoSeg")
    parser.add_argument("--isresize", choices=['True', 'False'], default='True', help="Specify if images should be resized to increase speed (output has original size again)")
    parser.add_argument("--resize_factor", type=int, default=0.5, help="Resize Factor. Height and width of images is multiplied by this factor if isresize = True")
    parser.add_argument("--size_thresh", type=int, default=2000, help="Threshold of minimum image size. If image is bigger than this threshold, it will be resized.")
    parser.add_argument("--area_thresh_ratio", type=int, default=0.01, help="Ratio that defines the minimun area a mask must have to be recognized (area_tresh_ratio = min_mask_area/total_image_area).")
    args = parser.parse_args()

    # Convert string to boolean (string is used for arg parsing)
    args.isresize = args.isresize == 'True'
    
    # Data path
    # Select your own data path! To try the script there are some images stored under "./imgs/set_1/"
    root_folder = "./imgs/set_1"
    # Path to store the segmented images
    destination_folder = "your-destination-path"

    # Model path
    ckpt_vit_b = "your-model-path/sam_vit_b_01ec64.pth"
    ckpt_vit_h = "your-model-path/sam_vit_h_4b8939.pth"

    # Init selected model
    sam = sam_model_registry["vit_b"](checkpoint=ckpt_vit_b)
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Iterate through folder
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        if os.path.exists(subdir_path):
            for image_name in os.listdir(subdir_path):
                print(datetime.now())
                segment_images(args, subdir_path, image_name, destination_folder)