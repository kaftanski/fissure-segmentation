#!/bin/bash
keypoints=("enhancement")
features=("image" "mind" "mind_ssc" "enhancement" "none")
gpu=2

for kp in "${keypoints[@]}"
  do
    if [[ "$kp" == "cnn" ]]; then
      features_cur=("${features[@]}" "$kp")
    else
      features_cur=("${features[@]}")
    fi

    for feat in "${features_cur[@]}"
      do
        if [[ "$feat" == "none" ]]; then
          cmd="python train.py --data fissures --ds ts --pts 2048 --k 40 --static --coords --batch 32 --gpu $gpu --output results/DGCNN_seg_$kp""_nofeat --kp_mode $kp --test_only"
        else
          if [[ "$feat" == "mind" ]] || [[ "$feat" == "mind_ssc" ]]; then
            cmd="python3.9 train.py --data fissures --ds ts --pts 2048 --k 40 --static --coords --batch 32 --gpu $gpu --output results/DGCNN_seg_$kp""_$feat --kp_mode $kp --patch $feat --test_only"
          else
            cmd="python3.9 train.py --data fissures --ds ts --pts 2048 --k 40 --static --coords --batch 32 --gpu $gpu --output results/DGCNN_seg_$kp""_$feat --kp_mode $kp --patch $feat"
          fi
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

cmd="python train.py --data fissures --ds ts --pts 2048 --k 40 --static --coords --batch 32 --gpu $gpu --output results/DGCNN_seg_foerstner_nofeat --kp_mode foerstner --test_only"

