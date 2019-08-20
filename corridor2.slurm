#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --p=short
#SBATCH --time=24:00:00
#SBATCH --begin=now
#SBATCH --job-name=exp-corridor
#SBATCH --mail-user=erwan.lecarpentier@isae-supaero.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

source venv/bin/activate
python corridor.py
deactivate
