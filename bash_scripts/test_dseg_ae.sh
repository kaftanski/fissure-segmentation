#!/bin/bash

run () {
  cmd="python3.9 test_ae_regularization.py --ds ts --gpu "$GPU" --output results/AE_"$OUT_SUFFIX"_"$1"_"$2" --seg_dir results/DGCNN_seg_"$1"_"$2" --ae_dir "$AE_DIR""
  echo "#######################################################################################################################################################################"
  echo $cmd
  echo "#######################################################################################################################################################################"
  $cmd
}

GPU=3
AE_DIR="results/pc_ae_regularized"
OUT_SUFFIX="reg"
keypoints=("cnn" "foerstner" "enhancement")
features=("image" "mind" "mind_ssc" "enhancement" "nofeat")

for kp in "${keypoints[@]}"
  do
    if [[ "$kp" == "cnn" ]]; then
      features_cur=("${features[@]}" "$kp")
    else
      features_cur=("${features[@]}")
    fi

    for feat in "${features_cur[@]}"
      do
        run "$kp" "$feat"
    done
  done
