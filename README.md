# Local policy search with Bayesian Optimization
We present a new method that allows us to use local gradient methods for black-box optimization. We actively sample new points in the Bayesian optimization (BO) framework for efficient gradient estimation and thus call our method Gradient Information with BO (GIBO).

All optimizers implemented in this repo can solve black-box optimization problems. 
Black-box optimization refers to the general setup of optimizing an unknown function where only its evaluations are available. 

# Code of the repo
- [optimizers](./src/optimizers.py): Implemented optimizers for black-box functions are [random search](https://arxiv.org/abs/1803.07055), vanilla Bayesian optimization, CMA-ES and the proposed method GIBO.
- [model](./src/model.py): A Gaussian process model with a squared-exponential kernel that also supplies its Jacobian.
- [policy parameterization](./src/policy_parameterizations.py): Multilayer perceptrones as policy parameterization for solving reinforcement learning problems.
- [environment api](./src/environment_api.py): Interface for interactions with reinforcement learning environments of OpenAI Gym.
- [acquisition function](./src/acquisition_function.py): Custom acquisition function for gradient information.
- [loop](./src/loop.py): Brings together all parts necessary for an optimization loop.


# Installation
Our GIBO implementation relies on mujoco-py 0.5.7 with MuJoCo Pro version 1.31.
To install MuJoCo follow the instructions here: [https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py).
To run Linear Quadratic Regulator experiments, follow the instruction under [gym-lqr](./gym-lqr/).

## Pip
Into an environment with python 3.8.5 you can install all needed packages with
```
pip install -r requirements.txt
```

## Conda
Or you can create an anaconda environment called gibo using
```
conda env create -f environment.yaml
conda activate gibo
```

## Pipenv
Or you can install and activate and environment via pipenv
```
pipenv install
pipenv shell
```

# Usage 
For experiments with synthetic test functions and reinforcement learning problems (e.g. MuJoCo) a command-line interface is supplied.

## Synthetic Test Functions
### Run
First generate the needed data for the synthetic test functions.

```
python generate_data_synthetic_functions.py -c ./configs/synthetic_experiment/generate_data_default.yaml
```

Afterwards you can run for instance our method GIBO on these test functions.

```
python run_synthetic_experiment.py -c ./configs/synthetic_experiment/gibo_default.yaml -cd ./configs/synthetic_experiment/generate_data_default.yaml
```

### Evaluate
Evaluation of the synthetic experiments and reproduction of the paper's figures can be done with the notebook [evaluation synthetic experiment](notebooks/evaluation_synthetic_experiment.ipynb).

## Reinforcement Learning
### Run
Run the MuJoCo swimmer environment with the proposed method GIBO.

```
python run_rl_experiment.py -c ./configs/rl_experiment/gibo_default.yaml
```

### Evaluate
Create plot to compare rewards over function calls for different optimizers (in this case gibo with random search).

```
python evaluation_rl_experiment.py -path path_to_image/image.pdf -cs ./configs/rl_experiment/gibo_default.yaml ./configs/rl_experiment/rs_default.yaml 
```
Or use the notebook [evaluation rl experiment](notebooks/evaluation_rl_experiment.ipynb) to reproduce the figures of the paper.

## Linear Quadratic Regulator
To reproduce the results and plots of the paper run the code in the notebook [lqr_experiment](notebooks/lqr_experiment.ipynb).

