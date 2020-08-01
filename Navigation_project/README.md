# Training an RL agent with the Double Deep Q-Network.

## The Environment

For this project, a Unity environment provided by Udacity is used to monitor and visualize training an RL agent. It is an open space with bananas of two different colours; blue and yellow.
The goal of the agent is to navigate through the environment picking as many yellow bananas and ignoring as many blue bananas as possible.

### The agent can perform 4 discrete actions to move;

- Move forward, 0
- Move backwards, 1
- Turn left, 2
- Turn right, 3

The environment also has a state-space of 37 dimensions which contains the agent’s velocity and a ray-based representation of objects around it.



## The Navigation_project content

In the navigation project folder, there are 4 sub-folders and 3 distinct files;

### The [Banana_Linux](https://github.com/Khaulat/Deep_Reinforcement_Learning/tree/master/Navigation_project/Banana_Linux) folder

Contains the Unity environment for training the agent. Depending on your operating system, you can download this environment from any of these links:

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- [Windows(32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

### The [python](https://github.com/Khaulat/Deep_Reinforcement_Learning/tree/master/Navigation_project/python) folder

Contains the python requirements for the Udacity-Unity environment as provided [here](https://github.com/udacity/deep-reinforcement-learning)

### The [NN_Model](https://github.com/Khaulat/Deep_Reinforcement_Learning/blob/master/Navigation_project/NN_Model.py) file

Contains the Deep Neural network used as function-approximator for the action-value function used to build the model.

### The [rl_algorithm](https://github.com/Khaulat/Deep_Reinforcement_Learning/blob/master/Navigation_project/rl_algorithm.py) file

Contains the reinforcement learning algorithm used - Double Deep Q-Network algorithm.

### The [run_navigation](https://github.com/Khaulat/Deep_Reinforcement_Learning/blob/master/Navigation_project/run_navigation.ipynb) file

With this file, we are able to run, train and visualize the complete navigation project.

### The [.ipynb_checkpoints](https://github.com/Khaulat/Deep_Reinforcement_Learning/tree/master/Navigation_project/.ipynb_checkpoints) folder

Contains the saved history of the files.

### The [__pycache__](https://github.com/Khaulat/Deep_Reinforcement_Learning/tree/master/Navigation_project/__pycache__)  folder

Contains the compiled bytecode of the Python source files.

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
    
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
    - Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
    - Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
    
3. Clone the repository (if you haven't already!), and navigate to the `Navigation_project/python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/Khaulat/Deep_Reinforcement_Learning.git
cd Deep_Reinforcement_learning/Navigation_project/python
pip install.
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.


### For a detailed technical understanding of the project, make sure to check out [this report](https://khaulat.github.io/Deep-Q-Networks(DQN)/).
