#!/bin/bash
#SBATCH --job-name=HSPIRESRTransformer
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=dmkoopmans@astro.rug.nl
#SBATCH --output=job-%j.log
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=4:00:00

module load Python/3.9.6-GCCcore-11.2.0
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1

python /home1/s3101940/Herschel-SRTransformer-inp4-2/train.py



