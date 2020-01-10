#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -n name -e n_epochs"
   echo -e "\t-n Name of experiment."
   echo -e "\t-e No. of epochs before decay."
   exit 1 # Exit script after printing help
}

while getopts "n:e:" opt
do
   case "$opt" in
      n ) name="$OPTARG" ;;
      e ) n_epochs="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

if [ -z "$name" ] || [ -z "$n_epochs" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

python train.py \
  --dataroot ./datasets/advnet_objects \
  --name $name \
  --gpu_ids 1 \
  --model advnet \
  --n_epochs $n_epochs \
  --display_id 0 \
  --save_epoch_freq 50
