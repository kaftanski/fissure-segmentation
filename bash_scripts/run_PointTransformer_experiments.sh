#!/bin/bash

run () {
  if [[ "$feat" == "nofeat" ]]; then
    cmd="python3.10 train.py --data fissures --ds ts --pts 2048 --coords --batch 32 --gpu "$gpu" --output results/PointTransformer_"$kp""_$feat" --kp_mode "$kp" --model PointTransformer"
  else
    cmd="python3.10 train.py --data fissures --ds ts --pts 2048 --coords --batch 32 --gpu "$gpu" --output results/PointTransformer_"$kp""_$feat" --kp_mode "$kp" --patch $feat --model PointTransformer"
  fi
  echo "#######################################################################################################################################################################"
  echo $cmd
  echo "#######################################################################################################################################################################"
  $cmd &
}

keypoints=("cnn" "foerstner" "enhancement")
features=("image" "mind" "mind_ssc" "enhancement" "nofeat")
gpus=( 2 3 )
gpu_index=0

for kp in "${keypoints[@]}"
  do
    if [[ "$kp" == "cnn" ]]; then
      features_cur=("${features[@]}" "$kp")
    else
      features_cur=("${features[@]}")
    fi

    for feat in "${features_cur[@]}"
      do
        gpu=${gpus[$gpu_index]}
        run "$kp" "$feat"

        gpu_index=$((gpu_index+1))
        gpu_index=$((gpu_index % ${#gpus[@]}))
        echo "GPU: $gpu"
    done
  done

wait
echo "All processes done!"
