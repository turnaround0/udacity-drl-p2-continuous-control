### Learning algorithm
Deep Deterministic Policy Gradient[(DDPG)](https://arxiv.org/abs/1509.02971) algorithm was used for this training.

### Hyper-parameters
```python
memory_params = {
    'buffer_size': int(1e6),        # replay buffer size
    'batch_size': 128,              # minibatch size
    'seed': 0,                      # Seed to generate random numbers
}
```

```python
params = {
    'gamma': 0.99,                      # discount factor
    'tau': 1e-3,                        # for soft update of target parameters
    'update_every': 1,                  # update parameters per this number
    'lr_actor': 7e-5,                   # learning rate of the Actor
    'lr_critic': 1e-4,                  # learning rate of the Critic
    'seed': 0,                          # Seed to generate random numbers
    'actor_units': [512, 256],          # Number of nodes in hidden layers of the Actor
    'critic_units': [512, 256],         # Number of nodes in hidden layers of the Critic
    'weight_decay': 0,                  # L2 weight decay
    'noise_theta': 0.15,                # Theta of Ornstein-Uhlenbeck process
    'noise_sigma': 0.01,                # Sigma of Ornstein-Uhlenbeck process
}
```

### model architectures

### Plot of Rewards
![](ddpg_plot.jpg)

### Ideas for Future Work
1. Test with multi-agents
It took too long time to train under multi-agents environment.<br>
[MMDDPG](https://arxiv.org/abs/1706.02275) can be used for the environment.

2. Try with another algorithms like PPO