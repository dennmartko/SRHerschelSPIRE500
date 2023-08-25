#!/bin/bash
#SBATCH --job-name=test-inference
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=dmkoopmans@astro.rug.nl
#SBATCH --output=job-%j.log
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
module load Python/3.9.6-GCCcore-11.2.0
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1

#export LD_LIBRARY_PATH=/lib:/software/software/Python/3.8.6-GCCcore-10.2.0/lib:/software/software/Python/3.8.6-GCCcore-10.2.0/lib64:/software/software/Python/3.8.6-GCCcore-10.2.0/bin
python /home1/s3101940/HERSPIRESRproj/PaperModel/test_GPU.py

