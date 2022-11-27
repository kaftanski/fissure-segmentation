#!/bin/bash

train () {
  cmd="python3.9 train.py --data fissures --ds ts --pts 2048 --coords --batch 32 --gpu 0 --model PointNet --output results/PointNet_seg_"$1"_"$2" --kp_mode "$1" --patch "$2" --train_only"
  echo "#######################################################################################################################################################################"
  echo $cmd
  echo "#######################################################################################################################################################################"
  $cmd
}

test () {
  cmd="/share/data_rechenknecht03_2/students/kaftan/FissureSegmentation/miniconda3/envs/fissure/bin/python3.9 train.py --gpu "$gpu" --output results/PointNet_seg_"$1"_"$2" --test_only"
  echo "#######################################################################################################################################################################"
  echo $cmd
  echo "#######################################################################################################################################################################"
  $cmd
}

gpu=0

#train "cnn" "image"
#train "enhancement" "image"
#train "foerstner" "image"

test "cnn" "image"
test "enhancement" "image"
test "foerstner" "image"
