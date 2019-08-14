#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --time=48:00:00
#SBATCH --begin=now
#SBATCH --mail-user=erwan.lecarpentier@isae-supaero.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=exp-corridor
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err
#SBATCH --partition=short

module purge
module load python/3.7
module list

SLURM_NODEFILE=my_slurm_node.$$
srun hostname | sort > $SLURM_NODEFILE
sort $SLURM_NODEFILE | uniq -c | xargs printf "(%d)    %s\n"

echo nbtask = $SLURM_NTASKS

source venv/bin/activate
python corridor.py
deactivate

rm $SLURM_NODEFILE
