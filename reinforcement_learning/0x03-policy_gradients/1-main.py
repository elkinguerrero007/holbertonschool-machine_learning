#!/usr/bin/env python3
"""
Main file
"""
import gym
import numpy as np
from policy_gradient import policy_gradient

env = gym.make('CartPole-v1')
np.random.seed(1)

weight = np.random.rand(4, 2)
state = env.reset()[None,:]
print(weight)
print(state)

result = policy_gradient(state, weight)
action = result[0]
grad = result[1]
print(action)
print(grad)

env.close()
