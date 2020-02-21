#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --partition=sgpu
#SBATCH --qos=normal
#SBATCH --output=unet-test-job-%j.out
#SBATCH --job-name=unet-test-job

module purge

source /curc/sw/anaconda3/2019.03/bin/activate
conda activate $PYTORCH
echo "== This is the scripting step! =="
sleep 30
python /projects/kadh5719/image-segmentation_antarctica/deep-learning/scripts/tester_unet.py
echo "== End of Job =="
