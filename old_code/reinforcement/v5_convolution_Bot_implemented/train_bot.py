from termcolor import colored

from chart import Trade_Env
import math
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from ai_memory import ReplayMemory
from ai_memory import Transition
import torch
from model_conv import DQN
import torch.optim as optim
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 5000
MIN_SAMPLES_TO_START_TRAINING = 10000
MEMORY_SIZE = 100000

MAX_EPSILON = 1.0
MIN_EPSILON = 0.05
LAMBDA = 0.001
GAMMA = 0.99

LEARNING_RATE = 0.0001

TARGET_UPDATE = 10

EPISODES = 5000


class TrainerBot:
    def __init__(self, input_rows, input_columns, num_outputs):
        self.h = input_rows
        self.w = input_columns
        self.o = num_outputs

        self._policy_net = None
        self._target_net = None
        self._memory = None
        self._optimizer = None

        self._episode_profits = []
        self._steps = 0

        self.reset()

    def reset(self):
        self._policy_net = DQN(self.h, self.w, self.o).to(device)
        self._target_net = DQN(self.h, self.w, self.o).to(device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        self._memory = ReplayMemory(MEMORY_SIZE)
        self._episode_profits = []
        self._steps = 0

        self._optimizer = optim.Adam(self._policy_net.parameters(), lr=LEARNING_RATE)

    def restore_checkpoint(self, path):
        if os.path.isfile(path):
            self._policy_net.load_state_dict(torch.load(path))
            self._target_net = DQN(self.h, self.w, self.o).to(device)
            self._target_net.load_state_dict(self._policy_net.state_dict())
            self._target_net.eval()

            print("Loaded checkpoint:", path)
        else:
            print(colored("Checkpoint" + path + "not found!", color="yellow"))

    def save_model(self, path):
        torch.save(self._target_net.state_dict(), path)

    def _select_action(self, state):
        sample = random.random()

        eps_threshold = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self._steps)
        if sample < eps_threshold:
            return torch.tensor([[random.randrange(self.o)]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self._policy_net(state).max(1)[1].view(1, 1)

    def _optimize_model(self):
        if len(self._memory) < MIN_SAMPLES_TO_START_TRAINING:
            return
        transitions = self._memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self._policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self._target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

    @property
    def episode_profits(self):
        return self._episode_profits

    def play_episode(self, env):
        self._steps += 1

        # Initialize the environment and state
        state = env.reset()

        print("\nEpisode:", self._steps, "      ", end="")

        eps_threshold = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self._steps)

        total_reward = 0

        gathering_samples_mode_active = len(self._memory) < MIN_SAMPLES_TO_START_TRAINING
        if gathering_samples_mode_active:
            print("Gathering samples ...", end="")
        else:
            print("Training!", end="")
        print("      eps:", eps_threshold)

        done = False
        while not done:
            # Select and perform an action
            action = self._select_action(state)
            next_state, reward, done = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)

            # Observe new state
            if done:
                next_state = None

            # Store the transition in memory
            self._memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
        self._optimize_model()

        if self._steps % TARGET_UPDATE == 0:
            self._target_net.load_state_dict(self._policy_net.state_dict())

        if not gathering_samples_mode_active:
            self.episode_profits.append(total_reward)

            print("Reward: ", total_reward)
            c = 'red' if total_reward < 0 else 'green'
            gx = 'X' * int(abs(total_reward))
            print(colored(gx, color=c))



