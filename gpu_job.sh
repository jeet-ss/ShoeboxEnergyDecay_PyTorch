#!/bin/bash -l 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=20:59:00 
#SBATCH --job-name=Arni_withoutNoise_0_83
#SBATCH --export=NONE 

unset SLURM_EXPORT_ENV 

# cpus-per-task has to be set again for srun
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

#module load python/3.10-anaconda
source $HOME/.rir/bin/activate
cd $HPCVAULT/ShoeboxEnergyDecay_PyTorch
#ls
#python --version
srun python train.py --ds 0 --de 83
