from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
from matplotlib import pyplot as plt
import numpy as np
import datetime
import os
import skimage
import skimage.measure

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

start_time = datetime.datetime.now()
# img_path = "C:\\Users\\faulhamm\\Documents\\270223_TF_C_W_DJI_0389_crop.jpg"
img_path = "C:/Users/stefa/Desktop/riccia_gan/riccia_imgs/Riccia bifurca/ABry_201_Riccia_bifurca_Spore_04.jpg"

root_folder = "C:/Users/stefa/Desktop/riccia_gan/riccia_imgs/"
destination_folder = "C:/Users/stefa/Desktop/riccia_gan/riccia_imgs_selected"

ckpt_vit_b = "C:/Users/stefa/Desktop/riccia_gan/sam_models/sam_vit_b_01ec64.pth"
ckpt_vit_h = "C:/Users/stefa/Desktop/riccia_gan/sam_models/sam_vit_h_4b8939.pth"


sam = sam_model_registry["vit_b"](checkpoint=ckpt_vit_b)
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

for subdir in os.listdir(root_folder):
    subdir_path = os.path.join(root_folder, subdir)
    if os.path.isdir(subdir_path) and "Riccia" in subdir:
        for image_name in os.listdir(subdir_path):
            #image = Image.open(subdir_path + '/' + image_name)
            image = cv2.imread(subdir_path + '/' + image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if not os.path.isdir(destination_folder + '/' + subdir):
                os.makedirs(destination_folder + '/' + subdir)
            #image.save(destination_folder + '/' + subdir + '/'  + image_name)


            masks = mask_generator.generate(image)

            for i in range(len(masks)):
                image_new_false = image.copy()
                bool_mask = masks[i]['segmentation']
                #labeled_image, count = skimage.measure.label(bool_mask, return_num=True)
                #object_features = skimage.measure.regionprops(labeled_image)
                #object_areas = [objf["area"] for objf in object_features]
                #for object_id, objf in enumerate(object_features, start=1):
                #    if objf["area"] < max(object_areas):
                #        labeled_image[labeled_image == objf["label"]] = False
                #    if objf["area"] == max(object_areas):
                #        labeled_image[labeled_image == objf["label"]] = True

                image_new_false[bool_mask == False] = [255,255,255]
                image_new_false = cv2.cvtColor(image_new_false, cv2.COLOR_RGB2BGR)
                if bool_mask[10,10] == False:
                    cv2.imwrite(destination_folder + '/' + subdir + '/'  + image_name + str(i) + 'true.png', image_new_false)
                '''
                image_new_true = image.copy()
                bool_mask = masks[i]['segmentation']
                labeled_image, count = skimage.measure.label(bool_mask, return_num=True)
                object_features = skimage.measure.regionprops(labeled_image)
                object_areas = [objf["area"] for objf in object_features]
                for object_id, objf in enumerate(object_features, start=1):
                    if objf["area"] < max(object_areas):
                        labeled_image[labeled_image == objf["label"]] = True
                    if objf["area"] == max(object_areas):
                        labeled_image[labeled_image == objf["label"]] = False
                
                image_new_true[bool_mask == True] = [255,255,255]
                image_new_true = cv2.cvtColor(image_new_true, cv2.COLOR_RGB2BGR)
                if bool_mask[10,10] == True:
                    cv2.imwrite(destination_folder + '/' + subdir + '/'  + image_name + str(i) + 'true.png', image_new_true)
                '''
end_time = datetime.datetime.now()

elapsed = end_time - start_time

plt.figure(figsize=(20,20))
plt.title("Time Elapsed: {0}".format(elapsed))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()



# predictor.set_image(image)
# masks, _, _ = predictor.predict("Tree")