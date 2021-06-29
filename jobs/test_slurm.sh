#!/bin/bash

#SBATCH -J test_job

#SBATCH -t 00:05:00

#SBATCH --partition=any_gpu

#SBATCH --cpus-per-task=1

echo ${SLURM_JOB_NAME} allocated to ${SLURM_NODELIST}
ls ..
