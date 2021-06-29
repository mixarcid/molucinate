#!/bin/bash
#SBATCH -J install
#SBATCH -t 5-00:30:00
#SBATCH --partition=any_gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

source /opt/anaconda3-cluster/etc/profile.d/conda.sh
module load cuda/11.1

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZjJlZWU5MjEtZTIzNS00NDU0LWFkNTEtOTY1NTNiZDVlMWVjIn0="

conda activate chem
cd ..

python train.py name=csb
