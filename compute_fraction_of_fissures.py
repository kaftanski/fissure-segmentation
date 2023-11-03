import pandas as pd
from tqdm import trange

from preprocess_totalsegmentator_dataset import TotalSegmentatorDataset
import SimpleITK as sitk
import numpy as np

ds = TotalSegmentatorDataset()
shape_stats_filter = sitk.LabelShapeStatisticsImageFilter()
fissure_sizes = pd.DataFrame(columns=['fissure_{}'.format(i) for i in range(1, ds.num_classes)] + ['all', 'total_size'], dtype=np.float32)

# iterate over all images
for i in trange(len(ds)):
    # compute label measures for fissure labels
    fissures = ds.get_regularized_fissures(i)
    shape_stats_filter.Execute(fissures)

    # get physical size of fissures
    new_row = {}
    total = 0
    for lbl in shape_stats_filter.GetLabels():
        new_row['fissure_{}'.format(lbl)] = shape_stats_filter.GetPhysicalSize(lbl)
        total += shape_stats_filter.GetPhysicalSize(lbl)
    new_row['all'] = total

    # compute total size of the image
    size = fissures.GetSize()
    spacing = fissures.GetSpacing()
    new_row['total_size'] = np.prod([sz * sp for sz, sp in zip(size, spacing)])

    # add new row to fissure sizes
    fissure_sizes.loc[i] = new_row

# compute fraction of fissures
fissure_sizes['fraction'] = fissure_sizes['all'] / fissure_sizes['total_size']

# compute mean and std of all columns
fissure_sizes.loc['mean'] = fissure_sizes.mean()
fissure_sizes.loc['std'] = fissure_sizes.std()

# save fissure sizes as csv
fissure_sizes.to_csv('./results/fissure_sizes.csv')
