#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -c class -e n_epochs -g gpu_id"
   echo -e "\t-c Name of ADE20K class to experiment on."
   echo -e "\t-e No. of epochs before decay."
   echo -e "\t-g GPU id to use."
   exit 1 # Exit script after printing help
}

while getopts "c:e:g:" opt
do
   case "$opt" in
      c ) class="$OPTARG" ;;
      e ) n_epochs="$OPTARG" ;;
      g ) gpu_id="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

if [ -z "$class" ] || [ -z "$n_epochs" ] || [ -z "$gpu_id" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

python train.py \
  --dataroot ./datasets/ADE20K \
  --name ade20k_$class \
  --gpu_ids $gpu_id \
  --model pix2pix \
  --dataset_mode ade20k \
  --dataset_class $class \
  --n_epochs $n_epochs \
  --display_id 0 \
  --save_epoch_freq 50
