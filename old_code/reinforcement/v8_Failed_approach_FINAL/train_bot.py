from termcolor import colored
import pandas as pd
from ai_player import play_chart
from chart import Trade_Env
import math
from ai_memory import ReplayMemory
from ai_memory import Transition
import torch
from model_conv import DQN
import torch.optim as optim
import torch.nn.functional as F
import os
import torch.multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainerBot:
    def __init__(self, params):
        self._movers = None

        path = params['chart_list_file']
        if os.path.isfile(path):
            self._movers = pd.read_csv(path)
        else:
            print("ERROR! Could not open file: ", path)

        self._batch_size = params['batch_size']
        self._gamma = params['gamma']

        env = Trade_Env(self._movers, simulation_mode=False)
        self.h, self.w = env.state_shape
        self.o = env.num_actions

        self._policy_net = None
        self._target_net = None

        self._memory = None
        self._optimizer = None

        self._episode_profits = []
        self._steps = 0

        self._memory = ReplayMemory(params['memory_size'])

        self.reset(params)

    def reset(self, params):

        self._policy_net = DQN(self.h, self.w, self.o).to(device)
        self._policy_net.share_memory()
        self._target_net = DQN(self.h, self.w, self.o).to(device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        self._episode_profits = []
        self._steps = 0

        self._optimizer = optim.Adam(self._policy_net.parameters(), lr=params['learning_rate'])

    def restore_checkpoint(self, path):
        res = False

        if os.path.isfile(path):
            self._policy_net.load_state_dict(torch.load(path))
            self._target_net = DQN(self.h, self.w, self.o).to(device)
            self._target_net.load_state_dict(self._policy_net.state_dict())
            self._target_net.eval()

            res = True

            print("Loaded checkpoint:", path)
        else:
            print(colored("Checkpoint" + path + "not found!", color="yellow"))

        return res

    def save_model(self, path):
        torch.save(self._target_net.state_dict(), path)

    def _optimize_model(self):
        if len(self._memory) < self._batch_size:
            return
        transitions = self._memory.sample(self._batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self._policy_net(state_batch).gather(1, action_batch)

        next_action_values = torch.zeros(self._batch_size, device=device)
        next_action_values[non_final_mask] = self._target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_action_values * self._gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

    @property
    def episode_profits(self):
        return self._episode_profits

    def train_episode(self, p):
        self._steps += 1

        eps = p['min_epsilon'] + (p['max_epsilon'] - p['min_epsilon']) * math.exp(-p['lambda'] * self._steps)

        gathering_samples_mode_active = len(self._memory) < p['batch_size']

        print("\nEpisode:", self._steps, "      ", end="")
        if gathering_samples_mode_active:
            print("Gathering samples ...", end="")
        else:
            print("Training!", end="")
        print("      eps:", eps)

        cpu_count = p['cpu_count']

        #######################################################################################
        # PLAY Section

        if cpu_count == 1:
            transactions, total_reward = play_chart(self._policy_net, eps, self._movers, p['sim_chart_index'])

            for t in transactions:
                self._memory.push(t)

            self.episode_profits.append(total_reward)
        else:
            pool = mp.Pool(cpu_count)
            mp_results = [pool.apply_async(play_chart, args=(self._policy_net, eps, self._movers)) for i in range(100)]
            pool.close()
            pool.join()

            for res in mp_results:
                transactions, total_reward = res.get(timeout=1)
                for t in transactions:
                    self._memory.push(t)

                self.episode_profits.append(total_reward)

        #######################################################################################
        # TRAINING Section

        self._optimize_model()

        if self._steps % p['steps_before_update'] == 0:
            self._target_net.load_state_dict(self._policy_net.state_dict())

        #######################################################################################

        print("Reward: ", total_reward)
        c = 'red' if total_reward < 0 else 'green'
        gx = 'X' * int(abs(total_reward))
        print(colored(gx, color=c))
