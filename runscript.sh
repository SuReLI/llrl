#!/bin/bash

find . -type f -name "slurm*" -exec sbatch {} \; 
