#!/bin/bash
keypoints=("cnn") #("foerstner" "cnn" "enhancement")
features=("image" "mind" "mind_ssc" "enhancement" "nofeat")
gpu=1

for kp in "${keypoints[@]}"
  do
    if [[ "$kp" == "cnn" ]]; then
      features_cur=("${features[@]}" "$kp")
    else
      features_cur=("${features[@]}")
    fi

    for feat in "${features_cur[@]}"
      do
        if [[ "$kp" == "foerstner" ]] && [[ "$feat" == "image" ]]; then
          continue
        fi

        cmd="python train.py --data fissures --ds ts --coords --gpu $gpu --output results/DGCNN_seg_$kp""_$feat --kp_mode $kp --patch $feat --test_only --copd"
        echo
        echo
        echo "####################################################################################################################################"
        echo Running $cmd
        echo "####################################################################################################################################"
        echo
        docker run --shm-size 1g -v /home/users/kaftan/FissureSegmentation:/opt/project --runtime=nvidia --rm --gpus all --detach --workdir /opt/project/fissure-segmentation fissure:latest $cmd
      done

    gpu=$((gpu+1))
  done
