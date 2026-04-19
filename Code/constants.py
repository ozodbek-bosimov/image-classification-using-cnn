import os

# Project root directory (.../image-classification-using-cnn)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
TEST_DIR = os.path.join(ROOT_DIR, 'test1')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
MODEL_DIR = os.path.join(ROOT_DIR, 'Code', 'models')

CAT_LBL = 0
DOG_LBL = 1
CAT = 'cat'
DOG = 'dog'
LABEL_MAP = {
    CAT: CAT_LBL,
    DOG: DOG_LBL
}
DATA_SIZE = 18_000
IMG_SIZE = 110
SPLIT_RATIO = 0.8