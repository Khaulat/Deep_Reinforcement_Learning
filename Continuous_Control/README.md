

 # Training an RL agent with the Deep Deterministic Policy Gradient Algorithm.

## The Environment

For this project, a Unity environment provided by Udacity is used to monitor and visualize training an RL agent. It to trains an agent in the form of a double-jointed arm to control a ball continuously.

To get a better result, the training is done in parallel by creating 20 copies of the same agent. Each copy learns and contributes its result to the main agent.

### The agent moves continuously;

Since the agent's actions is continuous, it is given a range between -1 and 1.


## The Continous_control project content

Some important folders and files in the Continous_Control project include;

### The [Reacher_Linux](hhttps://github.com/Khaulat/Deep_Reinforcement_Learning/tree/master/Continuous_Control/Reacher_Linux) folder

Contains the Unity environment for training the agent. Depending on your operating system, you can download this environment from any of these links:

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- [Windows(32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

### The [python](https://github.com/Khaulat/Deep_Reinforcement_Learning/tree/master/Continuous_Control/python) folder

Contains the python requirements for the Udacity-Unity environment as provided [here](https://github.com/udacity/deep-reinforcement-learning)

### The [model](https://github.com/Khaulat/Deep_Reinforcement_Learning/blob/master/Continuous_Control/model.py) file

Contains the Deep Neural network used as function-approximator for the action-value function used to build the model.

### The [ddpg_agent](https://github.com/Khaulat/Deep_Reinforcement_Learning/blob/master/Continuous_Control/ddpg_agent.py) file

Contains the reinforcement learning algorithm used - Deep Deterministic Policy Gradient algorithm.

### The [Control](https://github.com/Khaulat/Deep_Reinforcement_Learning/blob/master/Continuous_Control/Control.ipynb) file

With this file, we are able to run, train and visualize the complete continuous control project.


## Reproducing the project on your computer

Depending on what you want to do, [install Unity](https://github.com/Unity-Technologies/ml-agents/blob/release_3_docs/docs/Installation.md#advanced-local-installation-for-development) and follow one of these 3 options:

1. Clone the repository.
2. Fork the repository.
3. Generate a pull request to contribute your RL projects.

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

    - __Linux__ or __Mac__:
    ```bash
    conda create --name drlnd python=3.6
    source activate drlnd
    ```
    - __Windows__:
    ```bash
    conda create --name drlnd python=3.6
    activate drlnd
    ```
 
2. Clone the repository (if you haven't already!), and navigate to the `Continuous_Control/python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/Khaulat/Deep_Reinforcement_Learning.git
cd Deep_Reinforcement_learning/Continuous_Control/python
pip install.

```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.


### For a detailed technical understanding of the project, make sure to check out [this report](https://khaulat.github.io/Deep-Deterministic-Policy-Gradient/).

