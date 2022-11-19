import os

KP_MODES = ['foerstner', 'noisy', 'cnn', 'enhancement']
POINT_DIR = '/share/data_rechenknecht03_2/students/kaftan/FissureSegmentation/point_data'
POINT_DIR_TS = os.path.join(POINT_DIR, 'ts')

IMG_DIR = '/share/data_rechenknecht03_2/students/kaftan/FissureSegmentation/data'
IMG_DIR_TS = '/share/data_rechenknecht03_2/students/kaftan/FissureSegmentation/TotalSegmentator/ThoraxCrop_v2/'

DEFAULT_SPLIT = "../nnUNet_baseline/nnu_preprocessed/Task501_FissureCOPDEMPIRE/splits_final.pkl"
DEFAULT_SPLIT_TS = "../nnUNet_baseline/nnu_preprocessed/Task503_FissuresTotalSeg/splits_final.pkl"