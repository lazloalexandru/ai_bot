from env import Trade_Env
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
from model import DQN


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


print("CUDA Available: ", torch.cuda.is_available())

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

movers = pd.read_csv('data\\active_days.csv')

GAMMA = 0.99
MAX_EPSILON = 1.0
MIN_EPSILON = 0.1
LAMBDA = 0.001
TARGET_UPDATE = 10

env = Trade_Env(movers, simulation_mode=True, sim_chart_index=6611)

# Get number of actions from gym action space
n_actions = env.num_actions

_, h, w = env.state_shape
print("Input Size: ", h, w)

target_net = DQN(h, w, n_actions).to(device)
target_net.eval()


def select_action(s):
    return target_net(s).max(1)[1].view(1, 1)


episode_profits = []


def plot_durations():
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


train_steps = 5000
STEP = 500

num_episodes = int(train_steps / STEP)
for i_episode in range(1, num_episodes + 1):

    # if i_episode != 4:
    #    continue

    path = "checkpoints\\params_" + str(i_episode * STEP)
    print(path)

    target_net = DQN(h, w, n_actions).to(device)
    target_net.load_state_dict(torch.load(path))
    target_net.eval()

    # Initialize the environment and state
    env = Trade_Env(movers, simulation_mode=True, sim_chart_index=6511)
    state = env.reset()

    total_profit = 0

    print("\nEpisode:", i_episode)

    done = False
    while not done:
        # Select and perform an action
        action = select_action(state)
        next_state, reward, done = env.step(action.item())
        total_profit += reward
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if done:
            next_state = None

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)

    env.save_chart(str(i_episode * STEP))
    episode_profits.append(total_profit)

# plot_durations()

print('Complete')
# env.render()
# env.close()
# plt.show()
