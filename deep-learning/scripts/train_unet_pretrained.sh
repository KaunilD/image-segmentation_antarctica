#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --partition=sgpu
#SBATCH --qos=normal
#SBATCH --output=unet-train-job-%j.out
#SBATCH --job-name=unet-train-job

module purge

source /curc/sw/anaconda3/2019.03/bin/activate
conda activate $PYTORCH

echo "== This is the scripting step! =="

sleep 30
# lr-model-momentum-weightdecay
python /projects/kadh5719/image-segmentation_antarctica/deep-learning/scripts/train_unet_pretrained.py --lr 0.0001 --train-batch-size 64 --restart-checkpoint False --checkpoint-prefix unet-sgd-4-0.9-4

echo "== End of Job =="
