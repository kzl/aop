# Adaptive Online Planning

Associated code for our paper, Adaptive Online Planning for Continual Lifelong Learning. See our [website](https://sites.google.com/berkeley.edu/aop) for more details.

## Requirements

1. Clone/download a copy of this repository.
2. Code uses Python 3, as well as the following packages, which can be installed via pip/conda: numpy, gym, scipy, torch, matplotlib, and seaborn.
3. Install [MuJoCo](http://mujoco.org/) and [mujoco-py](https://github.com/openai/mujoco-py).

## Running Experiments

To run an experiment, run the command (all args are optional, use -h for help/more information):

```
python run.py --a aop -e hopper -s changing
```

## Visualizing Experiments

To visualize results, identify the directory of the experiment and run (replace ex/1124_1200 with relevant directory and 20000 with the length of the experiment):

```
python graph.py ex/1124_1200 20000
```
