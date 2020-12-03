from env import Trade_Env
import numpy as np
import pandas as pd
import common as cu
import matplotlib
import matplotlib.pyplot as plt
import torch
from model import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(nn, s):
    return nn(s).max(1)[1].view(1, 1)


def simulate_on_nn(path, params):
    gen_charts = 'gen_charts' in params.keys() and params['gen_charts']

    movers = pd.read_csv(params['active_days_file'])
    env = Trade_Env(movers,
                    simulation_mode=True,
                    sim_chart_index=params['sim_chart_idx'])
    n_actions = env.num_actions
    h, w = env.state_shape
    # print("Input Size: ", h, w)

    episode_rewards = []
    all_gains = []

    nn = Net(h, w, n_actions).to(device)
    nn.load_state_dict(torch.load(path))
    nn.eval()

    sim_chart_idx = None
    if 'sim_chart_idx' in params.keys():
        sim_chart_idx = params['sim_chart_idx']

    env = Trade_Env(movers,
                    simulation_mode=True,
                    sim_chart_index=sim_chart_idx)

    if sim_chart_idx is None:
        num_episodes = params['num_episodes']
    else:
        num_episodes = 1

    for i_episode in range(1, num_episodes + 1):
        print("\nEpisode:", i_episode)

        if sim_chart_idx is None:
            env.set_sim_chart_idx(params['base_chart_index'] + i_episode)

        state = env.reset()
        total_reward_per_episode = 0.0

        done = False
        while not done:
            # Select and perform an action
            action = select_action(nn, state).item()
            next_state, reward, gain, trade_done, done = env.step(action)

            if trade_done:
                all_gains.append(gain)

            total_reward_per_episode += reward
            reward = torch.tensor([reward], device=device)
            total_reward_per_episode += reward

            if done:
                next_state = None

            state = next_state

        if gen_charts:
            env.save_chart(str(params['checkpoint_id'] * params['checkpoint_steps']))

        episode_rewards.append(total_reward_per_episode)

    del nn
    del env

    return episode_rewards, all_gains


def simulate_checkpoints(params):
    train_steps = params['training_steps']
    STEP = params['checkpoint_steps']

    num_checkpoints = int(train_steps / STEP)
    print("Num Checkpoints:", num_checkpoints, "   ( Steps:", train_steps, "Interval:", STEP, ")")

    checkpoint_gains_list = []

    start_idx = 1
    if 'sim_last_checkpoint_only' in params.keys() and params['sim_last_checkpoint_only']:
        start_idx = num_checkpoints

    for checkpoint_id in range(start_idx, num_checkpoints+1):
        path = "checkpoints\\params_" + str(checkpoint_id * STEP)
        print(path)

        params['checkpoint_id'] = checkpoint_id

        _, checkpoint_gains = simulate_on_nn(path, params)

        checkpoint_gains_list.append(checkpoint_gains)

    #########################################
    # SHOW STATS

    show_header = True
    for gains in checkpoint_gains_list:
        cu.stats(gains,
                 show_header=show_header,
                 show_in_rows=True)

        show_header = False


def sim():

    params = {
        'active_days_file': 'data\\active_days.csv',

        'gen_charts': True,
        'num_episodes': 30,
        'sim_chart_idx': 5400,
        'base_chart_index': 5400,
        'training_steps': 4000,
        'checkpoint_steps': 1000,
        'sim_last_checkpoint_only': False
    }

    simulate_checkpoints(params)


sim()
