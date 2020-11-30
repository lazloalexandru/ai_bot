from termcolor import colored

from env import Trade_Env
import math
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from ai_memory import ReplayMemory
from ai_memory import Transition
import torch
from model import DQN
import torch.optim as optim
import torch.nn.functional as F
import os

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

print("CUDA Available: ", torch.cuda.is_available())

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

movers = pd.read_csv('data\\active_days.csv')
env = Trade_Env(movers, simulation_mode=False)

checkpoint_path = "checkpoints\\params_44000"

BATCH_SIZE = 10000
MIN_SAMPLES_TO_START_TRAINING = 20000
MEMORY_SIZE = 100000

MAX_EPSILON = 0.10001
MIN_EPSILON = 0.1
LAMBDA = 0.001
GAMMA = 0.99

TARGET_UPDATE = 10

EPISODES = 50000
SAVE_AT_STEP = 500

n_actions = env.num_actions
_, h, w = env.state_shape
print("Input Size: ", h, w)

policy_net = DQN(h, w, n_actions).to(device)

if os.path.isfile(checkpoint_path):
    policy_net.load_state_dict(torch.load(checkpoint_path))

target_net = DQN(h, w, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
memory = ReplayMemory(MEMORY_SIZE)


steps_done = 0
eps_threshold = 0


def select_action(state):
    sample = random.random()
    global eps_threshold

    eps_threshold = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps_done)
    if sample < eps_threshold:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)


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


def optimize_model():
    if len(memory) < MIN_SAMPLES_TO_START_TRAINING:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


eps_threshold = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps_done)


num_episodes = EPISODES + 1
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()

    steps_done += 1

    total_profit = 0

    print("Episode:", i_episode, "    ", end="")
    gathering_samples = len(memory) < MIN_SAMPLES_TO_START_TRAINING
    if gathering_samples:
        print("Gathering samples ...", end="")
    else:
        print("Training!", end="")
    print("      eps:", eps_threshold)

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

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
    optimize_model()

    if not gathering_samples:
        episode_profits.append(total_profit)

    plot_durations()

    print("Profit: ", total_profit)
    c = 'red' if total_profit < 0 else 'green'
    gx = 'X' * int(abs(total_profit))
    print(colored(gx, color=c))
    print("\n")

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if i_episode > 0 and i_episode % SAVE_AT_STEP == 0:
        path = "checkpoints\\params_" + str(i_episode)
        torch.save(target_net.state_dict(), path)

print('Complete')
# env.render()
# env.close()
plt.ioff()
plt.show()
