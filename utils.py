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
        os.mkdir(treated_images_directory)
    except OSError as error:
        ...

    annotation_files = list_all_files(annotation_files_directory)
    
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
                os.mkdir(os.path.join(treated_images_directory,annotation["label"]))
            except OSError as error:
                ...

            location = os.path.join(treated_images_directory, annotation["label"], sub_ref["sub_img_name"])
            
            img = crop_image(downloaded_image_name, annotation["coordinates"])
            img = fill_image(img)
            save_csv_image(location, img)

            ref["sub_images"].append(sub_ref)
            i += 1
            
        refs.append(ref)
    save_metadata(refs)
    save_labels(refs)
    ...



def crop_image(image_name, coordinates):
    image_path = os.path.join(os.getcwd(), downloaded_files_directory, image_name)
    img = cv2.imread(image_path)
    width = int(coordinates["width"])
    height = int(coordinates["height"])
    left = int(coordinates["x"] - width / 2)
    top = int(coordinates["y"] - height / 2)

    img = img[top:top+height, left:left+width]

    return img

def fill_image(cv2_img):     
    width, height, dims = cv2_img.shape
    
    if width > height:
        ratio = TRAIN_IMAGES_DIM / width
    else:
        ratio = TRAIN_IMAGES_DIM / height

    cv2_img = cv2.resize(cv2_img, None, fx=ratio, fy=ratio)

    width, height, dims = cv2_img.shape

    filler = None
    if width > height:
        filler = [[[255]*3]*(256 - height)] * 256
        cv2_img = np.append(cv2_img, filler, axis=1)
    else:
        filler = [[[255]*3] * 256] * (256 - width)
        cv2_img = np.append(cv2_img, filler, axis=0)
    return cv2_img
        

def save_csv_image(location, cv2_img): 
    cv2.imwrite(location, cv2_img)
    print("saved image ", location)