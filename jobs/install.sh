#!/bin/bash
#SBATCH -J install
#SBATCH -t 00:30:00
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err


#source /opt/anaconda3-cluster/etc/profile.d/conda.sh

#module load cuda/11.1
#conda activate chem

cd ..
ls

#cp cfg/platform/standard.yaml cfg/platform/local.yaml

#pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

#pip install -r requirements-pip.txt

cd ~/Data
mkdit mol-results
tar -xvf Zinc.tar.bz2

echo "Success!"
