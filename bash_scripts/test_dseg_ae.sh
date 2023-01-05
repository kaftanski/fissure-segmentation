#!/bin/bash

run () {
  cmd="python3.9 test_ae_regularization.py --ds ts --gpu "$GPU" --output results/DSEGAE_"$OUT_SUFFIX"_"$1"_"$2" --seg_dir results/DGCNN_seg_"$1"_"$2" --ae_dir "$AE_DIR""
  echo "#######################################################################################################################################################################"
  echo $cmd
  echo "#######################################################################################################################################################################"
  $cmd
}

GPU=2
AE_DIR="results/PC_AE_regularized_augment_1024"
OUT_SUFFIX="reg_aug_1024"
keypoints=("cnn" "foerstner" "enhancement")
#features=("image" "mind" "mind_ssc" "enhancement" "nofeat")

for kp in "${keypoints[@]}"
  do
    if [[ "$kp" == "cnn" ]]; then
      features_cur=("${features[@]}" "$kp")
    else
      features_cur=("${features[@]}")
    fi

    run "$kp" "image"
#    for feat in "${features_cur[@]}"
#      do
#        run "$kp" "$feat"
#    done
  done
