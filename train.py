from termcolor import colored
from env import Trade_Env
from train_bot import TrainerBot
import math
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from ai_memory import ReplayMemory
from ai_memory import Transition
import torch
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def plot_durations(episode_profits):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_profits, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Profit')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def do_training(num_episodes, save_step):
    plt.ion()

    checkpoint_path = "checkpoints\\params_11000"

    movers = pd.read_csv('data\\active_days.csv')
    env = Trade_Env(movers, simulation_mode=False)
    _, h, w = env.state_shape
    print("State Shape: ", h, w)

    bot = TrainerBot(h, w, env.num_actions)

    for i in range(1, num_episodes+1):
        bot.play_episode(env)

        plot_durations(bot.episode_profits)

        if i % save_step == 0:
            path = "checkpoints\\params_" + str(i)
            bot.save_model(path)

    print('Training Complete!')

    plt.ioff()
    plt.show()


do_training(num_episodes=5000, save_step=500)

