# Training RL agents with the Multi-Agent Deep Deterministic Policy Gradient Algorithm.

## The Environment

For this project, a Unity environment provided by Udacity is used to monitor and visualize training the RL agents. It trains multiple agents to play tennis with each other.



## The Cooperation and Competition project content

Some important folders and files in the Cooperation and Competition project include;

### The [Tennis_Linux]() folder

Contains the Unity environment for training the agent. Depending on your operating system, you can download this environment from any of these links:

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- [Windows(32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

### The [python](https://github.com/Khaulat/Deep_Reinforcement_Learning/tree/master/Cooperation%20and%20Competition/python) folder

Contains the python requirements for the Udacity-Unity environment as provided [here](https://github.com/udacity/deep-reinforcement-learning)

### The [model](https://github.com/Khaulat/Deep_Reinforcement_Learning/blob/master/Cooperation%20and%20Competition/model.py) file

Contains the Deep Neural network used as function-approximator for the action-value function used to build the model.

### The [maddpg_agent](https://github.com/Khaulat/Deep_Reinforcement_Learning/blob/master/Cooperation%20and%20Competition/maddpg_agent.py) file

Contains the reinforcement learning algorithm used - Multi-Agent Deep Deterministic Policy Gradient algorithm.

### The [Play_Tennis](https://github.com/Khaulat/Deep_Reinforcement_Learning/blob/master/Cooperation%20and%20Competition/Play_Tennis.ipynb) file

With this file, we are able to run, train and visualize the complete tennis project.


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
 
2. Clone the repository (if you haven't already!), and navigate to the `/Cooperation and Competition/python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/Khaulat/Deep_Reinforcement_Learning.git
cd Deep_Reinforcement_Learning/Cooperation and Competition/python/
pip install.

```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.


### For a detailed technical understanding of the project, make sure to check out [this report](https://khaulat.github.io/Multi-Agent-Reinforcement-Learning/).


