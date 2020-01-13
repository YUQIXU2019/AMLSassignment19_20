import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib

# PATH TO ALL IMAGES
global basedir, image_paths, target_size

basedir = '/Users/wyl/Desktop/AMLS_19-20_SN12345678/cartoon_set'
images_dir = os.path.join(basedir,'img')
labels_filename = 'new_labels.csv'

def extract_features_labels():
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extract the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    files = os.listdir(images_dir)
    files.sort(key = lambda x:int(x[:-4]))
    image_paths = [os.path.join(images_dir, l) for l in files]
    #image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    #target_size = None
    target_size = (256, 256, 1)
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    eye_color = {line.split(',')[0] : int(line.split(',')[1]) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for img_path in image_paths:
            file_name = img_path.split('.')[0].split('/')[-1]
            print(file_name)
            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            if img is not None:
                all_features.append(img)
                all_labels.append(eye_color[file_name+'.png'])

    landmark_features = np.array(all_features)
    eye_color = np.array(all_labels)  # simply converts the -1 into 0, so male=0 and female=1
    return landmark_features, eye_color