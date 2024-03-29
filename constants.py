import os

import matplotlib.cm

KP_MODES = ['foerstner', 'noisy', 'enhancement', 'cnn']
POINT_DIR = '../point_data'
POINT_DIR_TS = os.path.join(POINT_DIR, 'ts')

IMG_DIR = '../data'
IMG_DIR_TS = '../TotalSegmentator/ThoraxCrop_v2/'
IMG_DIR_TSv1 = '../TotalSegmentator/ThoraxCrop/'

DEFAULT_SPLIT = "../nnUNet_baseline/nnu_preprocessed/Task501_FissureCOPDEMPIRE/splits_final.pkl"
DEFAULT_SPLIT_TS = "../nnUNet_baseline/nnu_preprocessed/Task503_FissuresTotalSeg/splits_final.pkl"
FEATURE_MODES = ['mind', 'mind_ssc', 'image', 'enhancement', 'cnn']

CORR_FISSURE_LABEL_DEFAULT_TS = "./results/corresponding_points_ts/fissures/simple/labels.npz"
CLASSES = {1: 'LOF', 2: 'ROF', 3: 'RHF'}
CLASS_COLORS = ('r', matplotlib.cm.get_cmap('Set1').colors[2], 'dodgerblue')
