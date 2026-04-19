import numpy as np
import os 
from random import shuffle
import constants as CONST 
import cv2

def get_size_statistics():
    heights = []
    widths = []
    DIR = CONST.TRAIN_DIR
    for img in os.listdir(CONST.TRAIN_DIR):
        path = os.path.join(DIR, img)
        data = cv2.imread(path)
        if data is None:
            continue
        #data = np.array(Image.open(path))
        heights.append(data.shape[0])
        widths.append(data.shape[1])
    if not heights or not widths:
        print("No readable images found.")
        return
    avg_height = sum(heights) / len(heights)
    avg_width = sum(widths) / len(widths)
    print("Average Height: " + str(avg_height))
    print("Max Height: " + str(max(heights)))
    print("Min Height: " + str(min(heights)))
    print('\n')
    print("Average Width: " + str(avg_width))
    print("Max Width: " + str(max(widths)))
    print("Min Width: " + str(min(widths)))

#get_size_statistics()


def label_img(name):
    word_label = name.split('.')[0].lower()
    if word_label not in CONST.LABEL_MAP:
        return None
    label = CONST.LABEL_MAP[word_label]
    label_arr = np.zeros(2, dtype=np.float32)
    label_arr[label] = 1
    return label_arr


def prep_and_load_data():
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
        print(count)
        if count >= CONST.DATA_SIZE:
            break

    shuffle(data)

    #with open('train_data.pickle', 'wb') as train_d_file:
    #    pickle.dump(train_data, train_d_file)
    print(len(data))
    print('done')

    return data


if __name__ == "__main__":
    prep_and_load_data()
    



