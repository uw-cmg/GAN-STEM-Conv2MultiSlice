#!/usr/bin/env bash

# Ron the short-list GPU queue
#SBATCH -p slurm_sbel_cmg
#SBATCH --account=cmg --qos=cmg_owner

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH -t 4-1:00 # time (D-HH:MM)

## Create a unique output file for the job
#SBATCH -o cuda_Training-%j.log

## Load custimized CUDA
module load usermods
module load user/cuda
#module load cuda/8.0
# activate virtual environment
source activate pix

# install tensorflow and other libraries for machine learning
# All software installnation should be outside the scripts

conda install --name pix keras 

/srv/home/shenmr/anaconda3/envs/pix/bin/pip git+https://www.github.com/keras-team/keras-contrib.git

conda install --name pix matplotlib 
conda install --name pix numpy 
conda install --name pix scipy 
conda install --name pix pillow
conda install --name pix scikit-image
#conda install --name pix keras-contrib

# run the training scripts
python pix2pix.py
