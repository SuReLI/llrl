# Lipschitz Lifelong Reinforcement Learning

Value transfer experiments leveraging Lipschitz continuity of the optimal Q value function across MDPs.

## Use

The code is provided with a virtual environment including all the dependencies.
In order to use this virtual environment, you need to run the following command from this directory:

    virtualenv venv --distribute
    source venv/bin/activate
    
From there, you can run the script using the embedded python version.

## Experiments

To run the experiments of the Lipschitz Lifelong Reinforcement Learning paper, go to the experiments repository and run the following scripts:

Experiment 1:

	python bounds_comparison.py

Experiment 2:

	python prior use.py
	
Experiment 3:

	python lifelong_corridor.py
	python lifelong_maze_mono.py
	python lifelong_maze_multi.py

