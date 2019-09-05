import sys


def slurm_from_script_name(script_name):
    return [
        '#!/bin/bash',
        '',
        '#SBATCH --nodes=1',
        '#SBATCH --ntasks-per-node=24',
        '#SBATCH --ntasks-per-node=24',
        '#SBATCH --partition=short',
        '#SBATCH --time=24:00:00',
        '#SBATCH --mail-user=erwan.lecarpentier@isae-supaero.fr',
        '#SBATCH --mail-type=FAIL,END',
        '#SBATCH --job-name=' + script_name,
        '#SBATCH --output=slurm.%j.out',
        '#SBATCH --error=slurm.%j.err',
        '',
        'module purge',
        'module load python/3.7',
        '',
        'SLURM_NODEFILE=my_slurm_node.$$',
        'srun hostname | sort > $SLURM_NODEFILE',
        r'sort $SLURM_NODEFILE | uniq -c | xargs printf "(%d)    %s\n"',
        '',
        'echo nbtask = $SLURM_NTASKS',
        '',
        'source deactivate',
        'source activate myenv',
        'python ' + script_name + '.py',
        'source deactivate',
        '',
        'rm $SLURM_NODEFILE'
    ]


def generate_from_script_name(script_name):
    file_name = script_name + '.slurm'
    content = slurm_from_script_name(script_name)
    f = open(file_name, "w+")
    for row in content:
        f.write(row + '\n')
    f.close()


def remove_py(s):
    return s if s[-3:] != '.py' else s[:-3]


def generate(script_list):
    for s in script_list:
        generate_from_script_name(remove_py(s))


if __name__ == '__main__':
    generate(sys.argv[1:])
