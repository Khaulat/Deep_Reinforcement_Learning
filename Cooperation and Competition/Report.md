## Understanding Multi-Agent Reinforcement Learning

This concept comes from the fact that most agents don't exist alone. Instead, they interact, collaborate and compete with each other. Multi-Agent RL is bringing multiple single-agent together which can still retain their individual actions and rewards or have joint actions and rewards. There are various forms of environments in which agents can interact with each other;

- **Cooperative environments** - This is when all the agents involved have a common goal. Therefore, they cooperate to achieve them.
- **Competitive environments** - This is when the agents involved have separate goals. They compete with each other in order to meet their individual goals and maximize their individual reward.
- **Mixed Cooperative-Competitive environments** - This is when agent's act both cooperatively and competitively in order to maximize the overall reward.

Two reinforcement learning agents to play tennis. As in real tennis, the goal of each player is to keep the ball in play. And, when there're two equally matched opponents, their exchanges tend to be longer - where the hit the ball back and forth more frequently and accurately over the net.


## Learning Algorithm

The algorithm used here is an extension of the Deep Deterministic Policy Gradient as used [here](https://khaulat.github.io/Deep-Deterministic-Policy-Gradient/). The Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm

I was able to get a better prediction by making these improvements;


**`Fixed targets`** : Having fixed targets helps to stabilize training. Since we are using two neural networks for the actor and the critic, we have two targets, one for actor and critic each.

**`Gradient Clipping`** : Clipping the gradient so it doesn't overshoot. I implemented it using the *`torch.nn.utils.clip_grad_norm_`* function. This placed an upper limit on the size of the parameter updates and preventing them from growing exponentially.

**`Experience Replay`** : In order to take the time to learn every bit of an agent's observation which cannot be done at once, we store these observations in a replay buffer and learn from them from time to time by selecting the at random. This way, we're replaying the experiences.


## Methods

The **`Play_Tennis.ipynb`** contains the code for training and evaluation, **`model.py`** contains the Q-Network architecture while **`maddpg_agent.py`** contains the reinforcement learning algorithm. All contained in [this repository](https://github.com/Khaulat/Deep_Reinforcement_Learning/tree/master/Cooperation%20and%20Competition).

After tuning, the hyperparameters that prove to fit the model well are;

| Hyperparameter | Value |
| ----------- | ----------- |
| BUFFER_SIZE | 1e6 |
| BATCH_SIZE  | 128 |
| GAMMA | 0.99 |
| LR_actor | 1e-3 |
| LR_critic | 1e-3 |
| TAU | 8e-3|
| epsilon_start | 5.0 |
| epsilon_end | 300 |
| ou_sigma | 0.2 |
| ou_theta | 0.15 |

Some new terms were introduced here; the `OU_SIGMA` and `OU_THETA`. These represent the Ornstein–Uhlenbeck process, a stationary Gauss–Markov process which means that it is a Gaussian process(Markov process), and is temporally homogeneous. It is the only process that satisfies the condition of being Markov and homogeneous, and also allows for linear transformations of space and time variables. Here, we use them as **`noise parameters`**.

## Results

****

The DDPG agent was able to solve the environment giving the result below. The corresponding model weights are stored in Play_Tennis-checkpoint.ipynb.

<img src="/madresult.png" alt="Results of the agent learning from MADDPG">


## Some Future Improvements

The algorithm used has been able to learn well, however, other methods can still be explored;

- **Tuning hyperparameters**: We can try to tune the hyperparameters more.

- **Using DD4G**: This algorithm has in some cases given better results than DDPG. When implemented here, it might perform better.


## References

- [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments)
- [Issues in Using Function Approximation for Reinforcement Learning](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
