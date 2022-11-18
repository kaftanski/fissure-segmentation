#!/bin/bash
keypoints=("cnn" "enhancement")
features=("image" "mind" "mind_ssc" "none")
gpu=2

for kp in "${keypoints[@]}"
  do
    if [[ "$kp" == "cnn" ]] || [[ "$kp" == "enhancement" ]]; then
      features_cur=("${features[@]}" "$kp")
    fi

    for feat in "${features_cur[@]}"
      do
        if [[ "$feat" == "none" ]]; then
          cmd="python train.py --data fissures --ds ts --pts 2048 --k 40 --static --coords --batch 32 --gpu $gpu --output results/DGCNN_seg_$kp""_nofeat --kp_mode $kp"
        else
          cmd="python train.py --data fissures --ds ts --pts 2048 --k 40 --static --coords --batch 32 --gpu $gpu --output results/DGCNN_seg_$kp""_$feat --kp_mode $kp --patch $feat"
        fi
        echo
        echo
        echo "#######################################################################################################################################################################"
        echo $cmd
        echo "#######################################################################################################################################################################"
        echo
        $cmd
      done
  done
