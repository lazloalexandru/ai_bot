from termcolor import colored

from env import Trade_Env
import math
import random
from ai_memory import ReplayMemory
from ai_memory import Transition
import torch
from model import DQN
import torch.optim as optim
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(state, max_actions, nn, eps):
    sample = random.random()

    if sample < eps:
        return torch.tensor([[random.randrange(max_actions)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            return nn(state).max(1)[1].view(1, 1)


def play_chart(nn, eps, movers):
    env = Trade_Env(movers, simulation_mode=False)
    state = env.reset()

    total_reward = 0
    memory = []

    done = False
    while not done:
        action = select_action(state, env.num_actions, nn, eps)
        next_state, reward, _, _, done = env.step(action.item())

        total_reward += reward

        if done:
            next_state = None

        t_reward = torch.tensor([reward], device=device)
        memory.append(Transition(state, action, next_state, t_reward))

        state = next_state

    return memory, total_reward




