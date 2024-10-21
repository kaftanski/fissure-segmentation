# Sparse Keypoint Segmentation of Lung Fissures: Efficient Geometric Deep Learning for Abstracting Volumetric Images
This repository contains the code for segmenting pulmonary fissures in CT images using keypoint extraction and geometric deep learning.

# Usage
## Install environment
Install the required packages using Anaconda and the provided environment file:
```bash
conda env create -f environment.yml
conda activate fissure-segmentation
```

# Data
Datasets used in the evaluation are openly available for research purposes. Please refer to the original sources 
detailed below.

## Training / cross-validation data
The TotalSegmentator data set is the main data set used for this work. 
Download it from here: https://doi.org/10.5281/zenodo.6802614 (v1.0).
The data has to be preprocessed with `preprocess_totalsegmentator_dataset.py`, make sure to point to the correct data
path `TS_RAW_DATA_PATH` in `constants.py`. The script selects applicable images (showing the entire thorax), extracts fissure 
segmentations from the lobe masks and crops the images to the thoracic region. Results will be written to 
`IMG_DIR_TS_PREPROC`.

## COPD validation data set
The COPD data set can be accessed here: https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/copdgene.html
(citation: Castillo et al., Phys Med Biol 2013 https://doi.org/10.1088/0031-9155/58/9/2861)
The manual fissure annotations are available from here: http://www.mpheinrich.de/research.html#COPD (citation: 
Rühaak et al., IEEE TMI 2017 https://doi.org/10.1109/TMI.2017.2691259).

## Pre-computing keypoints and features
The image data needs to be preprocessed by extracting keypoints and point features. Images need to be in one folder
with names of the form `<patid>_img_<fixed/moving>.nii.gz` with the corresponding fissure annotations 
`<patid>_fissures_<fixed/moving>.nii.gz` and lung mask `<patid>_mask_<fixed/moving>.nii.gz`. Masks can be generated
automatically by, e.g., using the `lungmask` tool https://github.com/JoHof/lungmask. Please set the image directory path
`IMG_DIR` in `constants.py`.

Run the following scripts:
- TODO
- ...

# Citation
Please cite the following papers if you use parts of this code in your own work:

- Journal paper for Point Cloud Autoencoder (PC-AE) and comparison of geometric segmentation networks (under review):
```
@article{Kaftan2024_FissureSegmentation_IJCARS,
  title={Sparse Keypoint Segmentation of Lung Fissures: Efficient Geometric Deep Learning for Abstracting Volumetric Images},
  author={Kaftan, Paul and Heinrich, Mattias P and Hansen, Lasse and Rasche, Volker and Kestler, Hans A and Bigalke, Alexander},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  year={tbd},
  volume={tbd},
  number={tbd},
  pages={tbd},
  doi={tbd}
}
```

- Conference paper for comparison of more keypoints and features:
```
@InProceedings{Kaftan2024_FissureSegmentation_BVM,
  author = {Kaftan, Paul and Heinrich, Mattias P. and Hansen, Lasse and Rasche, Volker and Kestler, Hans A. and Bigalke, Alexander},
  editor = {Maier, Andreas and Deserno, Thomas M. and Handels, Heinz and Maier-Hein, Klaus and Palm, Christoph and Tolxdorff, Thomas},
  title = {Abstracting Volumetric Medical Images with Sparse Keypoints for Efficient Geometric Segmentation of Lung Fissures with a Graph CNN},
  booktitle = {Bildverarbeitung f{\"u}r die Medizin 2024},
  year = {2024},
  publisher = {Springer Fachmedien Wiesbaden},
  address = {Wiesbaden},
  pages = {60--65},
  isbn = {978-3-658-44037-4},
  doi = {10.1007/978-3-658-44037-4_19}
}
```
