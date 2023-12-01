# use-segment-anything-model-to-autosegment-microscope-images


This is a code implementation of Facebooks "Segment Anything Model"
A sample script how to utilize the Segment Anything Model to automatically segment microscopy images of spores

A detailed description was published on Medium: 
https://medium.com/@stefan.herdy/use-meta-ais-segment-anything-model-to-autosegment-microscopy-images-9a286837c56b

# Table of Contents

    Installation
    Usage
    License

Installation

```shell

$ git clone https://github.com/stefanherdy/use-segment-anything-model-to-autosegment-microscope-images.git
```
# Usage

-  First, download the desired Segment Anything Models from the original GitHub page:
   https://github.com/facebookresearch/segment-anything
-  Copy the models to your model folder
-  Set your data path, your destination path and your model path and run the script sam.py
- Optionally, you can run postprocess.py to crop the selected masks and delete mask artifacts to keep the biggest one
- The script classify.py is runs a pytorch classifier. The images and the regarding classes have to be in the subdirectories of ./imgs/ with set_1/, set_2/, set_3/ etc.

- Run sam.py with "python train.py".
    You can specify the following parameters:
    --isresize", type=bool, default=False, help="Specify if images should be resized to increase speed (output has original size again)"
    --resize_factor", type=int, default=0.5, help="Resize Factor. Height and width of images is multiplied by this factor if isresize = True"
    --area_thresh_ratio", type=int, default=0.01, help="Ratio that defines the minimun area a mask must have to be recognized (area_tresh_ratio = min_mask_area/total_image_area)."

        Example usage:
        "python3 sam.py --isresize True --resize_factor 0.5 --area_thresh_ratio 0.005

# License

This project is licensed under the MIT License.
:copyright: 2023 Stefan Herdy
