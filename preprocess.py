import os
from PIL import Image

def my_function(folder_path):
    # replace this with the code that you want to run on each subfolder
    print(f"Processing folder: {folder_path}")

root_folder = "C:/Users/stefa/Desktop/riccia_gan/data/raw"
destination_folder = "C:/Users/stefa/Desktop/riccia_gan/riccia_imgs"
for subdir in os.listdir(root_folder):
    subdir_path = os.path.join(root_folder, subdir)
    if os.path.isdir(subdir_path) and "Riccia" in subdir:
        for image_name in os.listdir(subdir_path):
            if "proximal" in image_name:
                image = Image.open(subdir_path + '/' + image_name)
                if not os.path.isdir(destination_folder + '/' + subdir):
                    os.makedirs(destination_folder + '/' + subdir)
                image.save(destination_folder + '/' + subdir + '/'  + image_name)

