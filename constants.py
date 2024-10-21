import os

import matplotlib.cm

KP_MODES = ['foerstner', 'noisy', 'enhancement', 'cnn']

DATA_DIR = 'data'

TS_RAW_DATA_PATH = '../TotalSegmentator/Totalsegmentator_dataset/'

IMG_DIR = os.path.join(DATA_DIR, 'images')
IMG_DIR_TS_PREPROC = os.path.join(IMG_DIR, 'TotalSegmentator')
IMG_DIR_COPD = os.path.join(IMG_DIR, 'COPD')

POINT_DIR = os.path.join(DATA_DIR, 'points')
POINT_DIR_TS = os.path.join(POINT_DIR, 'TotalSegmentator')
POINT_DIR_COPD = os.path.join(POINT_DIR, 'COPD')

DEFAULT_SPLIT = "../nnUNet_baseline/nnu_preprocessed/Task501_FissureCOPDEMPIRE/splits_final.pkl"
DEFAULT_SPLIT_TS = "../nnUNet_baseline/nnu_preprocessed/Task503_FissuresTotalSeg/splits_final.pkl"
FEATURE_MODES = ['mind', 'mind_ssc', 'image', 'enhancement', 'cnn']

KEYPOINT_CNN_DIR = "results/lraspp_recall_loss"

CLASSES = {1: 'LOF', 2: 'ROF', 3: 'RHF'}
CLASS_COLORS = ('r', matplotlib.cm.get_cmap('Set1').colors[2], 'dodgerblue')

ALIGN_CORNERS = False
