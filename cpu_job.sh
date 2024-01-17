#!/bin/bash -l 
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00 
#SBATCH --job-name=gen_data
#SBATCH --export=NONE 

unset SLURM_EXPORT_ENV 

# cpus-per-task has to be set again for srun
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

#module load python/3.10-anaconda
source $HOME/.gen/bin/activate
cd $HPCVAULT/ShoeboxEnergyDecay_PyTorch
#
srun python generate_data.py