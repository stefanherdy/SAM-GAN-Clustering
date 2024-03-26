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
-  Copy the models to your model folder.
-  Set your data path, your destination path and your model path and run the script sam.py.
- Optionally, you can run postprocess.py to crop the selected masks and delete mask artifacts to keep the biggest one.

- How to run sam.py with "python sam.py":

        You can specify the following parameters:
        --isresize, type=bool, default=False, help="Specify if images should be resized to increase speed (output has original size again)"
        --resize_factor, type=int, default=0.5, help="Resize Factor. Height and width of images is multiplied by this factor if isresize = True"
        --size_thresh, type=int, default=2000, help="Threshold of minimum image size. If image is bigger than this threshold, it will be resized."
        --area_thresh_ratio, type=int, default=0.01, help="Ratio that defines the minimun area a mask must have to be recognized (area_tresh_ratio = min_mask_area/total_image_area)."

        Example usage:
        python sam.py --isresize True --resize_factor 0.5 --area_thresh_ratio 0.005

The code in this repository was used for further analysis of microscope images of different species. The aim was to investigate how well Generative Adversarial Networks (GANs) can assist in species recognition.

- gan.py and gan_128.py execute a Generative Adversarial Network to generate artificial images based on the selected images.
- The script classify.py runs a PyTorch classifier to compare how effectively real and generated images can be classified. This enables conclusions to be drawn regarding how well the GAN recognizes and generates class-specific features, as good recognition and generation of class-specific features should increase the accuracy of classification of the generated data compared to the real data. The images and their corresponding classes should be located in subdirectories of ./imgs/, such as set_1/, set_2/, set_3/, and so on.
- read_records.py performs a statistical analysis of the different tests.

# License

This project is licensed under the MIT License.
©️ 2023 Stefan Herdy
