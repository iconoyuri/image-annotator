import os
import json
import numpy as np
from PIL import Image
import cv2

from config import *


def get_directory_path(directory):
    parent_directory = os.getcwd()
    path = os.path.join(parent_directory, directory)
    return path

def list_all_files(directory):
    _directory = get_directory_path(directory)
    entries = os.walk(_directory)
    file_list = []
    for entry in entries:
        file_list += [os.path.join(f"{entry[0]}/", file) for file in entry[2]]
    return file_list

def save_metadata(refs):
    with open(reference_file, "w") as f:
        f.write(json.dumps(refs, indent=4))

def save_labels(refs):
    labels = []
    for ref in refs:
        for sub_ref in ref["sub_images"]:
            labels.append(sub_ref["label"])

    labels = list(dict.fromkeys(labels))
    with open(os.path.join( labels_file), "w") as f:
        for label in labels:
            f.write(label + '\n')

def load_metadata(file_name):
    with open(file_name) as f:
        return json.load(f)

def convert_image_to_csv(image_path, save_folder):
    if os.path.splitext(image_path)[1] in SUPPORTED_FORMATS :

        image = Image.open(image_path)
        
        arr = np.asarray(image)
        lst = []
        for row in arr:
            tmp = []
            for col in row:
                tmp.append(str(col))
            lst.append(tmp)

        csv_file = os.join(save_folder, f"{os.path.splitext(image_path)[0]}.csv")
        with open(csv_file, 'w') as f:
            for row in lst:
                f.write(','.join(row) + '\n')
        return csv_file
    return None


def crop_all_files():
    try:
        os.mkdir(cropped_images_directory)
    except OSError as error:
        ...

    annotation_files = list_all_files(annotation_files_directory)
    # print(annotation_files)
    refs = []
    i = 0
    for file in annotation_files:

        if not file.endswith(".json"):
            continue

        metadata = load_metadata(file)[0]
        downloaded_image_name = metadata["image"]

        if not os.path.splitext(downloaded_image_name)[1] in SUPPORTED_FORMATS :
            continue

        ref = {}
        ref["annotation_file"] = file
        ref["downloaded_image"] = downloaded_image_name
        ref["sub_images"] = []

        for annotation in metadata["annotations"]:
            sub_ref = {}
            sub_ref["label"] = annotation["label"]
            sub_ref["sub_img_name"] = f"SUBIMG-{i}{os.path.splitext(downloaded_image_name)[1]}"
            try:
                os.mkdir(os.path.join(cropped_images_directory,annotation["label"]))
            except OSError as error:
                ...
            image = crop_image(downloaded_image_name, annotation["coordinates"])
            image.save(os.path.join(cropped_images_directory,annotation["label"], sub_ref["sub_img_name"]))

            ref["sub_images"].append(sub_ref)
            i += 1
            
        refs.append(ref)
    save_metadata(refs)
    save_labels(refs)
    ...



def crop_image(image_name, coordinates):
    image_path = os.path.join(os.getcwd(), downloaded_files_directory, image_name)
    img = Image.open(image_path)
    width = coordinates["width"]
    height = coordinates["height"]
    left = coordinates["x"] - width / 2
    right = coordinates["x"] + width / 2
    top = coordinates["y"] - height / 2
    bottom = coordinates["y"] + height / 2
    img = img.crop((left,top,right,bottom))

    # if width > height:
    #     ratio = TRAIN_IMAGES_DIM / width
    # else:
    #     ratio = TRAIN_IMAGES_DIM / height

    # img = cv2.resize(img, None, ratio, ratio)

    return img

def resize_images():
    images_list = list_all_files(cropped_images_directory)
    for image in images_list:
        if not os.path.splitext(image)[1] in SUPPORTED_FORMATS :
            continue

        img = cv2.imread(image)        
        width, height, dims = img.shape
        
        if width > height:
            ratio = TRAIN_IMAGES_DIM / width
        else:
            ratio = TRAIN_IMAGES_DIM / height

        img = cv2.resize(img, None, fx=ratio, fy=ratio)

        width, height, dims = img.shape

        filler = None
        if width > height:
            filler = [[[255]*3]*(256 - height)] * 256
            img = np.append(img, filler, axis=1)
        else:
            filler = [[[255]*3] * 256] * (256 - width)
            img = np.append(img, filler, axis=0)
            
        save_csv_image(image, img)

def save_csv_image(image, cv2_img):    
    try:
        os.mkdir(img_csv_file_directory)
    except OSError as error:
        ...
    location = os.path.splitext(os.path.basename(image))[0]
    location = f"{os.path.join(img_csv_file_directory, location)}.jpg"
    
    cv2.imwrite(location, cv2_img)
    print("saved image ", location)