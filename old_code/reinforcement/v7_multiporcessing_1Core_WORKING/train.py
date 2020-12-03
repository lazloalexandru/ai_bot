from chart import Trade_Env
from train import TrainerBot
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch

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


def do_training(params):
    plt.ion()

    bot = TrainerBot(params)
    bot.restore_checkpoint(params['restore_checkpoint'])

    num_episodes = params['num_episodes']
    save_step = params['save_step']

    for episode_id in range(1, num_episodes+1):
        bot.train_episode(params)

        plot_durations(bot.episode_profits)

        if episode_id % save_step == 0:
            path = "checkpoints\\params_" + str(episode_id)
            bot.save_model(path)

    print('Training Complete!')

    plt.ioff()
    plt.show()


def get_params():
    params = {
        'max_epsilon': 1.0,
        'min_epsilon': 0.05,
        'lambda':  0.01,

        'num_episodes': 5000,
        'save_step': 100,

        'memory_size': 100000,
        'batch_size': 10000,
        'play_batch_size': 100,
        'cpu_count': 2,


        'chart_list_file': 'data\\active_days.csv',
        'restore_checkpoint': 'checkpoints\\params_98000'
    }
    return params


if __name__ == '__main__':
    do_training(get_params())

