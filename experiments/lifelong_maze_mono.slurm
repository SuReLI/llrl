#!/bin/bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=24
#SBATCH --time=72:00:00
#SBATCH --begin=now
#SBATCH --mail-user=erwan.lecarpentier@isae.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=exp
#SBATCH -o    slurm.%j.out
#SBATCH -e    slurm.%j.err
#SBATCH --partition=medium

module purge
module load python/3.7
module list

SLURM_NODEFILE=my_slurm_node.$$
srun hostname | sort > $SLURM_NODEFILE
sort $SLURM_NODEFILE | uniq -c | xargs printf "(%d)    %s\n"

echo nbtask = $SLURM_NTASKS

source venv/bin/activate
python lifelong_maze_mono.py
deactivate

rm $SLURM_NODEFILE
