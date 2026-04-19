import numpy as np
import os
from random import shuffle
import constants as CONST
import cv2


def get_size_statistics():
    """Print height/width statistics of training images."""
    heights = []
    widths = []
    DIR = CONST.TRAIN_DIR
    for img in os.listdir(DIR):
        path = os.path.join(DIR, img)
        data = cv2.imread(path)
        if data is None:
            continue
        heights.append(data.shape[0])
        widths.append(data.shape[1])
    if not heights or not widths:
        print("No readable images found.")
        return
    print(f"Average Height: {sum(heights) / len(heights):.1f}")
    print(f"Max Height: {max(heights)}, Min Height: {min(heights)}")
    print(f"Average Width: {sum(widths) / len(widths):.1f}")
    print(f"Max Width: {max(widths)}, Min Width: {min(widths)}")
    print(f"Total images: {len(heights)}")


def label_img(name):
    """Convert filename to one-hot label array."""
    word_label = name.split('.')[0].lower()
    if word_label not in CONST.LABEL_MAP:
        return None
    label = CONST.LABEL_MAP[word_label]
    label_arr = np.zeros(2, dtype=np.float32)
    label_arr[label] = 1
    return label_arr


def prep_and_load_data():
    """Load, resize, normalize images and return list of [image, label] pairs."""
    DIR = CONST.TRAIN_DIR
    data = []
    image_paths = os.listdir(DIR)
    shuffle(image_paths)
    count = 0
    for img_path in image_paths:
        label = label_img(img_path)
        if label is None:
            continue
        path = os.path.join(DIR, img_path)
        image = cv2.imread(path)
        if image is None:
            continue
        image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))
        image = image.astype('float32') / 255.0
        data.append([image, label])
        count += 1
        if count % 1000 == 0:
            print(f"{count} images loaded...")
        if count >= CONST.DATA_SIZE:
            break

    shuffle(data)
    print(f"Total loaded: {len(data)} images")
    return data


if __name__ == "__main__":
    prep_and_load_data()
