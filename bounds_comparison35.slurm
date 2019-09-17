#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --ntasks-per-node=24
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --mail-user=erwan.lecarpentier@isae-supaero.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=bounds_comparison35
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

module purge
module load python/3.7

SLURM_NODEFILE=my_slurm_node.$$
srun hostname | sort > $SLURM_NODEFILE
sort $SLURM_NODEFILE | uniq -c | xargs printf "(%d)    %s\n"

echo nbtask = $SLURM_NTASKS

source deactivate
source activate myenv
python bounds_comparison.py 35
source deactivate

rm $SLURM_NODEFILE
