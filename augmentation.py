import cv2
import random
import numpy as np
from utils import list_all_files, save_csv_image
from config import *
import os

brightness_prefix = "br"
channel_shift_prefix = "cs"
horizontal_flip_prefix = "hf"
vertical_flip_prefix = "vf"
rotation_prefix = "rp"

def images_augmentation():
    treated_images = list_all_files(treated_images_directory)
    for image in treated_images:
        root_name = os.path.splitext(image)[0]
        root_extension = os.path.splitext(image)[1]
        img = cv2.imread(image)

        augmented_image_path = root_name + "(" + brightness_prefix + ")" + root_extension
        augmented_image = brightness(img, 0.5, 3)
        save_csv_image(augmented_image_path, augmented_image)


        augmented_image_path = root_name + "(" + channel_shift_prefix + ")" + root_extension
        augmented_image = channel_shift(img, 60)
        save_csv_image(augmented_image_path, augmented_image)

        augmented_image_path = root_name + "(" + horizontal_flip_prefix + ")" + root_extension
        augmented_image = horizontal_flip(img)
        save_csv_image(augmented_image_path, augmented_image)

        augmented_image_path = root_name + "(" + vertical_flip_prefix + ")" + root_extension
        augmented_image = vertical_flip(img)
        save_csv_image(augmented_image_path, augmented_image)

        for angle in [30,-30,45,-45]:
            augmented_image_path = f"{root_name}({rotation_prefix})({angle}){root_extension}"
            augmented_image = rotation(img, angle)
            save_csv_image(augmented_image_path, augmented_image)
        ...
    ...


def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def channel_shift(img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    return img

def horizontal_flip(img):
    return cv2.flip(img, 1)

def vertical_flip(img):
    return cv2.flip(img, 0)

def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img