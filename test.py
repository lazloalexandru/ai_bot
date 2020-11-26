import pandas as pd
from env import Trade_Env
import numpy as np


def test1():
    movers = pd.read_csv('data\\active_days.csv')
    env = Trade_Env(movers)

    s = env.reset()
    s, _, _ = env.step(2)
    s, _, _ = env.step(2)
    s, _, _ = env.step(0)
    s, _, _ = env.step(0)
    s, _, _ = env.step(2)
    s, _, _ = env.step(1)
    s, _, _ = env.step(1)
    s, _, _ = env.step(1)

    s, _, _ = env.step(1)
    s, _, _ = env.step(0)
    print(s.shape)

    z = np.reshape(s, (7, 390))
    print(z)


def test2():
    movers = pd.read_csv('data\\active_days.csv')
    env = Trade_Env(movers)

    z = env.reset()
    print(z)
    for i in range(0, 395):
        print("")
        s, _, _ = env.step(0)
        print(s.shape)
        z = np.reshape(s, (7, 390))
        print(z)


test2()
