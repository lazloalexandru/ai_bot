from chart import Trade_Env
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
from model import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(nn, s):
    return nn(s).max(1)[1].view(1, 1)


def simulate():
    movers = pd.read_csv('data\\active_days.csv')
    env = Trade_Env(movers, simulation_mode=True, sim_chart_index=6611)
    n_actions = env.num_actions
    _, h, w = env.state_shape
    print("Input Size: ", h, w)

    episode_profits = []

    train_steps = 5000
    STEP = 500

    num_episodes = int(train_steps / STEP)
    for i_episode in range(1, num_episodes + 1):
        # if i_episode != 4:
        #    continue

        path = "checkpoints\\params_" + str(i_episode * STEP)
        print(path)

        nn = DQN(h, w, n_actions).to(device)
        nn.load_state_dict(torch.load(path))
        nn.eval()

        env = Trade_Env(movers, simulation_mode=True, sim_chart_index=6511)
        state = env.reset()

        total_profit = 0

        print("\nEpisode:", i_episode)

        done = False
        while not done:
            # Select and perform an action
            action = select_action(nn, state)
            next_state, reward, done = env.step(action.item())
            total_profit += reward
            reward = torch.tensor([reward], device=device)
            total_profit += reward

            if done:
                next_state = None

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)

        env.save_normalized_chart(str(i_episode * STEP))
        episode_profits.append(total_profit)

        del nn
        del env

    print('Complete')


simulate()
