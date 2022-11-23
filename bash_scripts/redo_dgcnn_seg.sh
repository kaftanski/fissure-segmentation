#!/bin/bash
gpu=2

train () {
  if [[ "$2" == "none" ]]; then
    cmd="python3.9 train.py --data fissures --ds ts --pts 2048 --k 40 --static --coords --batch 32 --gpu "$gpu" --output results/DGCNN_seg_"$1"_nofeat --kp_mode "$1" --train_only"
  else
    cmd="python3.9 train.py --data fissures --ds ts --pts 2048 --k 40 --static --coords --batch 32 --gpu "$gpu" --output results/DGCNN_seg_"$1"_"$2" --kp_mode "$1" --patch "$2" --train_only"
  fi
  echo "#######################################################################################################################################################################"
  echo $cmd
  echo "#######################################################################################################################################################################"
  $cmd
}

test () {
  if [[ "$2" == "none" ]]; then
    cmd="python3.9 train.py --gpu "$gpu" --output results/DGCNN_seg_"$1"_nofeat --test_only"
  else
    cmd="python3.9 train.py --gpu "$gpu" --output results/DGCNN_seg_"$1"_"$2" --test_only"
  fi
  echo "#######################################################################################################################################################################"
  echo $cmd
  echo "#######################################################################################################################################################################"
  $cmd
}


train "enhancement" "none"
train "enhancement" "enhancement"
train "cnn" "mind_ssc"
train "foerstner" "none"

# TODO:
# test "cnn" "enhancement"
# test "cnn" "image"
# test "cnn" "none"
# test "enhancement" "none"
# test "enhancement" "enhancement"
# test "cnn" "mind_ssc"
# test "foerstner" "none"
