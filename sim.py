import os
import numpy as np
import tensorflow.compat.v1 as tf
from termcolor import colored
import pandas as pd
from model import Model
from env import Trade_Env
import train

tf.compat.v1.disable_eager_execution()


class SimBot:
    def __init__(self, sess, model, env, model_path):
        self._sess = sess
        self._env = env
        self._model = model
        self._model_path = model_path.replace("checkpoints\\", "")
        self._steps = 0
        self._reward_store = []
        self._max_x_store = []

    def run(self):
        state = self._env.reset()
        tot_reward = 0

        while True:
            action = self._choose_action(state)
            next_state, reward, done = self._env.step(action)

            if done:
                next_state = None

            state = next_state
            tot_reward += reward

            if done:
                c = 'red' if tot_reward < 0 else 'green'
                print(colored("Account: %.2f" % tot_reward, color=c), '\n')
                # self._env.save_traded_chart(self._model_path)
                self._env.save_traded_chart()
                self._reward_store.append(tot_reward)
                break

        return tot_reward

    def _choose_action(self, state):
        return np.argmax(self._model.predict_one(state, self._sess))

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def max_x_store(self):
        return self._max_x_store


def test():
    __active_days_file = "data\\active_days.csv"

    if not os.path.isfile(__active_days_file):
        print(colored("ERROR: " + __active_days_file + " not found!", color="red"))
    else:
        movers = pd.read_csv(__active_days_file)

        tr = Trade_Env(movers,
                       sim_chart_index=2300,
                       simulation_mode=True)

        model = Model(tr.num_states, tr.num_actions, train.get_params())

        with tf.Session() as sess:
            if model.restore(sess):
                bot = SimBot(sess, model, tr, model.path)
                num_episodes = 1
                cnt = 0
                while cnt < num_episodes:
                    print('Episode {} of {}'.format(cnt+1, num_episodes))
                    bot.run()
                    cnt += 1
                # cu.stats(bot.reward_store)


if __name__ == "__main__":
    test()
