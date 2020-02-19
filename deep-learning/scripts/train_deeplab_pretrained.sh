#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --partition=sgpu
#SBATCH --qos=normal
#SBATCH --output=semseg-train-job-%j.out
#SBATCH --job-name=semseg-train-job

module purge

source /curc/sw/anaconda3/2019.03/bin/activate
conda activate $PYTORCH
echo "== This is the scripting step! =="
sleep 30
python "/projects/kadh5719/image-segmentation_antarctica/deep-learning/scripts/train_deeplab_pretrained.py
  --lr 0.001 --train-batch-size 64
  --restart-checkpoint True
  --checkpoint-path /projects/kadh5719/image-segmentation_antarctica/deep-learning/models/session_0/deeplabv3_pretrained---bn2d-38.pth
  --checkpoint-prefix deeplabv3-s0-38-lr-3
  "
echo "== End of Job =="
