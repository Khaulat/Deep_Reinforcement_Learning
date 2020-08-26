## Understanding Deep Deterministic Policy Gradients

This is another type of deep reinforcement learning algorithm which combines both policy-based methods and value-based methods. It belongs to the actor-critic family of RL models. I use this algorithm in this project to train an agent in the form of a double-jointed arm to control a ball continuously. This action is important, especially in robotics.

Like any other typical reinforcement learning problem, the 'environment', 'state', 'rewards', 'observations' and 'actions' need to be defined.

- **Environment** - This is the area where the agent navigated through and learns. For this project, a Unity environment is used.
- **State** - The current status of the agent. The agent could be in any of the states at any time.
- **Observation** - This is what the agent sees in an environment. Here, the agents’ observation space consists of 33 variables corresponding to the position, rotation, velocity, and the angular velocity of the arm.
- **Action** - Whatever step the agent takes in a given state. For this project, the agent has an action range of -1 to 1.
- **Reward** - The prize got for taking any action; whether right or wrong. The agent gets a **+0.1** reward for every time it's in contact with the ball.

The training was done in parallel in order to maximize learning and make it faster. For this, 20 copies of the same agents learnt how to control the ball.

![The agent in it's environment](https://github.com/Khaulat/khaulat.github.io/blob/master/images/reacher.gif)


## Learning Algorithm

This algorithm as mentioned above is an actor-critic method which combines both value-based methods and policy-based method in order to get a better prediction. Policy-based methods like REINFORCE, have the problem of high variance while value-based methods like DQN have the problem of high bias. The combination of the two helps them complement each other.

The actor is a neural network which updates the policy and the critic is another neural network which evaluates the policy being learned which is, in turn, used to train the actor. This is a form of check and balance where the critic criticizes the actions of the actor and makes sure they are right.

The Deep Deterministic Policy Gradient was used here and as used in [this DQN Project](https://khaulat.github.io/Deep-Q-Networks(DQN)/), I was able to get a better prediction by making these improvements;


**`Fixed targets`** : Having fixed targets helps to stabilize training. Since we are using two neural networks for the actor and the critic, we have two targets, one for actor and critic each.

**`Soft Updates`** : The target networks are updated using soft updates where during each update step, 0.01% of the local network weights are mixed with the target networks weights, i.e. 99.99% of the target network weights are retained and 0.01% of the local network’s weights are added.

**`Experience Replay`** : In order to take the time to learn every bit of an agent's observation which cannot be done at once, we store these observations in a replay buffer and learn from them from time to time by selecting the at random. This way, we're replaying the experiences.


Parameters used to store and sample from the replay buffer include;

- **`BUFFER_SIZE`** '1e6' - This is the size of the replay buffer. Specifies how much experiences should be stored.
- **`BATCH_SIZE`** '256' - The number of experiences we take from the replay buffer at a time to train the agent.
- **`UPDATE_FACTOR`** '10' - The specifies the number of timesteps it takes to sample a batch from the replay buffer.
- **`LR_actor`** '1e-3'- The speed at which the agent learns from experiences for the actor.
- **`LR_critic`** '1e-3'- The speed at which the agent learns from experiences for the critic.

As mentioned above, DDPG uses a different Network for selecting actions and a different one for evaluating. Two Q-Networks(Local and Target Networks) with the same architecture but different weights serve this purpose.
The local network is trained by minimizing the mean-squared loss function and used the Adam optimizer with a learning rate of **LR**, then the target network is updated(*soft update*) using the value of **TAU** towards the local network.


## Methods

The **`Control.ipynb`** contains the code for training and evaluation, **`model.py`** contains the Q-Network architecture while **`ddpg_agent.py`** contains the reinforcement learning algorithm. All contained in [this repository](https://github.com/Khaulat/Deep_Reinforcement_Learning/tree/master/Continuous_Control).

After tuning, the hyperparameters that prove to fit the model well are;

| Hyperparameter | Value |
| ----------- | ----------- |
| BUFFER_SIZE | 1e6 |
| BATCH_SIZE  | 256 |
| UPDATE_FACTOR | 10 |
| GAMMA | 0.99 |
| LR_actor | 1e-3 |
| LR_critic | 1e-3 |
| TAU | 1e-3|
| epsilon | 1.0 |
| epsilon_decay | 1e-6 |


## Results

The DDPG agent was able to solve the environment after 174 episodes with an average score of +30, however, there's still room for improvements! The corresponding model weights are stored in ./checkpoint_actor.pth and ./checkpoint_critic.pth respectively.  

<img src="{{ site.url }}{{ site.baseurl }}/images/ddpg_result.png" alt="Results of the agent learning from DDPG">


## Some Future Improvements

The algorithm used has been able to learn well, however, other methods can still be explored;

- **Tuning hyperparameters**: We can try to tune the hyperparameters more.

- **Using DD4G**: This algorithm has in some cases given better results than DDPG. When implemented here, it might perform better.


## References

- [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- [Issues in Using Function Approximation for Reinforcement Learning](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
 
