import os
import numpy as np
import tensorflow.compat.v1 as tf
from termcolor import colored
import matplotlib.pylab as plt
import random
import math
import pandas as pd
from model import Model
from ai_memory import Memory
from train_env import Trade_Env

tf.compat.v1.disable_eager_execution()


class TrainerBot:
    def __init__(self, sess, model, env, memory, params, render=True):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory

        self._params = params

        self._eps = self._params['MAX_EPSILON']
        self._MAX_EPSILON = self._params['MAX_EPSILON']
        self._MIN_EPSILON = self._params['MIN_EPSILON']
        self._LAMBDA = self._params['LAMBDA']
        self._GAMMA = self._params['GAMMA']

        self._steps = 0
        self._reward_store = []
        self._performance_store = []

        self._render = render

    def run(self, full_predict_mode=False):
        state = self._env.reset()
        tot_reward = 0

        if self._steps > self._params['TRAINING_START']:
            if full_predict_mode:
                print("Measurement Step:", self._steps, "Eps:", self._eps)
            else:
                print("Training Active. Step:", self._steps, "Eps:", self._eps)
        else:
            print("Random Simulation. Step:", self._steps, "Eps:", self._eps)

        while True:
            action = self._choose_action(state, full_predict_mode)
            next_state, reward, done = self._env.step(action)

            if done:
                next_state = None

            if not full_predict_mode:
                self._memory.add_sample((state, action, reward, next_state))

            if self._steps > self._params['TRAINING_START'] and not full_predict_mode:
                if self._steps % self._params['TRAINING_STEP'] == 0:
                    self._replay()

            # exponentially decay the eps value
            if not full_predict_mode:
                self._steps += 1

            self._eps = self._MIN_EPSILON + (self._MAX_EPSILON - self._MIN_EPSILON) * math.exp(-self._LAMBDA * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                if full_predict_mode:
                    self._performance_store.append(tot_reward)
                    print("Performance:", tot_reward, self._performance_store[-1])
                else:
                    self._reward_store.append(tot_reward)
                break

        if full_predict_mode:
            gx = 'X' * int(abs(tot_reward))
            c = 'red' if tot_reward < 0 else 'green'
            print("Account: ", colored("%.2f" % tot_reward, color=c))
            print(colored(gx, color=c))
            print("\n")
        else:
            gx = '*' * int(abs(tot_reward))
            c = 'red' if tot_reward < 0 else 'green'
            print("Account: ", colored("%.2f" % tot_reward, color=c))
            print(colored(gx, color=c))
            print("\n")

        return tot_reward

    def _choose_action(self, state, full_prediction_mode):
        if full_prediction_mode:
            return np.argmax(self._model.predict_one(state, self._sess))
        else:
            if random.random() < self._eps:
                return random.randint(0, self._model.num_actions - 1)
            else:
                return np.argmax(self._model.predict_one(state, self._sess))

    def _replay(self):
        batch = self._memory.sample(self._model.batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model.num_states) if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._model.num_states))
        y = np.zeros((len(batch), self._model.num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + self._GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def performance_store(self):
        return self._performance_store


def train(params):

    active_days_file = params['active_days_file']
    if not os.path.isfile(active_days_file):
        print(colored("ERROR: " + active_days_file + " not found!", color="red"))
    else:

        movers = pd.read_csv(active_days_file)

        tr = Trade_Env(movers)

        print(tr.num_actions, tr.num_states)

        model = Model(tr.num_states, tr.num_actions, params)
        mem = Memory(params['MEMORY'])

        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:
            model.restore(sess)

            bot = TrainerBot(sess, model, tr, mem, params, False)
            num_episodes = params['EPISODES']
            cnt = 0
            while cnt < num_episodes:
                print('Episode {} of {}'.format(cnt+1, num_episodes))
                bot.run()
                cnt += 1
                if cnt % params['CHECKPOINT_AT_EPISODE_STEP'] == 0:
                    model.save(sess, cnt)

                # if cnt % params['STATS_PER_STEP'] == 0:
                #    bot.run(full_predict_mode=True)

            print(bot.performance_store)
            print(bot.reward_store)
            plt.plot(bot.performance_store)
            plt.show()
            plt.plot(bot.reward_store)
            plt.show()

            plt.close("all")


def get_params():
    params = {
        'MAX_EPSILON': 1.00,
        'MIN_EPSILON': 0.05,
        'LAMBDA': 0.00001,

        'GAMMA': 0.99,
        'EPISODES': 1000,
        'CHECKPOINT_AT_EPISODE_STEP': 50,
        'MAX_CHECKPOINTS': 50,
        'BATCH_SIZE': 200,
        'TRAINING_START': 500,
        'TRAINING_STEP': 10,
        'MEMORY': 50000,
        'STATS_PER_STEP': 50,
        'active_days_file': "data\\active_days.csv"
    }

    return params


if __name__ == "__main__":
    train(get_params())
