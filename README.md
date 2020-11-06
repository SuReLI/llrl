# Lipschitz Lifelong Reinforcement Learning

Value transfer experiments leveraging Lipschitz continuity of the optimal Q value function across MDPs.

## Use

The code is provided with a virtual environment including all the dependencies.
In order to use this virtual environment, you need to run the following command from this directory:

    source activate [absolute-path-to-this-repo]/venv
    
To deactivate:

    source deactivate
    
From there, you can run the script using the embedded python version.

## Experiments

To run the experiments of the Lipschitz Lifelong Reinforcement Learning paper, go to the experiments repository and run the following scripts:

Experiment 1:

	python tight.py

Experiment 2:

	python bounds_comparison.py
	
## Additional experiments

Additional experiments on the corridor, maze and heat-map environments can be found in the following scripts:

	experiments/lifelong_corridor.py
	experiments/lifelong_maze_mono.py
	experiments/lifelong_maze_multi.py
	experiments/lifelong_heat_map.py

