import pandas as pd
from tqdm import trange

from preprocess_totalsegmentator_dataset import TotalSegmentatorDataset
import SimpleITK as sitk
import numpy as np

ds = TotalSegmentatorDataset()
shape_stats_filter = sitk.LabelShapeStatisticsImageFilter()
fissure_sizes = pd.DataFrame(columns=['fissure_{}_mm3'.format(i) for i in range(1, ds.num_classes)] +
                                     ['all_mm3', 'total_size_mm3'] +
                                     ['fissure_{}_n_vox'.format(i) for i in range(1, ds.num_classes)] +
                                     ['all_n_vox', 'total_size_n_vox'], dtype=np.float32)

# iterate over all images
for i in trange(len(ds)):
    # compute label measures for fissure labels
    fissures = ds.get_regularized_fissures(i)
    shape_stats_filter.Execute(fissures)

    # get physical size of fissures
    new_row = {}
    total_mm3 = 0
    total_n_vox = 0
    for lbl in shape_stats_filter.GetLabels():
        new_row['fissure_{}_mm3'.format(lbl)] = shape_stats_filter.GetPhysicalSize(lbl)
        new_row['fissure_{}_n_vox'.format(lbl)] = shape_stats_filter.GetNumberOfPixels(lbl)
        total_mm3 += shape_stats_filter.GetPhysicalSize(lbl)
        total_n_vox += shape_stats_filter.GetNumberOfPixels(lbl)

    new_row['all_mm3'] = total_mm3
    new_row['all_n_vox'] = total_n_vox

    # compute total size of the image
    size = fissures.GetSize()
    spacing = fissures.GetSpacing()
    new_row['total_size_n_vox'] = np.prod(size)
    new_row['total_size_mm3'] = np.prod([sz * sp for sz, sp in zip(size, spacing)])

    # add new row to fissure sizes
    fissure_sizes.loc[i] = new_row

# compute fraction of fissures
fissure_sizes['fraction_n_vox'] = fissure_sizes['all_n_vox'] / fissure_sizes['total_size_n_vox']
fissure_sizes['fraction_mm3'] = fissure_sizes['all_mm3'] / fissure_sizes['total_size_mm3']

# compute mean and std of all columns
fissure_sizes.loc['mean'] = fissure_sizes.mean()
fissure_sizes.loc['std'] = fissure_sizes.std()

# save fissure sizes as csv
fissure_sizes.to_csv('./results/fissure_sizes.csv')
