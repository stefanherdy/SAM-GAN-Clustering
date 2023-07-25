import os
import cv2
import glob

root_folder = "C:/Users/stefa/Desktop/repos/use-segment-anything-model-to-autosegment-microscope-images/riccia_imgs_selected/"
destination_folder = "C:/Users/stefa/Desktop/repos/use-segment-anything-model-to-autosegment-microscope-images/riccia_imgs_checked"

for subdir in os.listdir(root_folder):
    subdir_path = os.path.join(root_folder, subdir)
    output_subdir = os.path.join(destination_folder, subdir)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    for file in os.listdir(subdir_path):
        new_name = os.path.join(destination_folder, subdir, file)
        # Load the image from the given path
        image_path = os.path.join(root_folder, subdir, file)
        image = cv2.imread(image_path)
        ex = glob.glob(os.path.join(new_name))
        if len(ex) > 0:
            print('Image already processed!')
        if len(ex) == 0:

            # Display the image
            image_cropped = cv2.resize(image, (512,512))
            cv2.imshow("Image", image_cropped)

            while True:
                # Wait for a key press and store the ASCII value of the key
                key = cv2.waitKey(0)

                # Check if the key is 'd' (ASCII value 100)
                if key == 100:
                    break
                if key != 100:
                    cv2.imwrite(new_name, image)
                    break