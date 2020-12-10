import numpy as np
import matplotlib.pyplot as plt
import os


def show_batch_stats(learning_rate):
    batch_sizes = [16, 32, 64, 128, 256, 512, 1024, 4096]

    for batch_size in batch_sizes:
        path = "log\\" + str(batch_size) + "_" + str(learning_rate) + ".dat"
        if os.path.isfile(path):
            loss = np.fromfile(path)
            loss = np.clip(loss, 0, 3)
            plt.plot(loss, label="%s Batch_Size" % batch_size)

    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.legend()
    plt.title("Learning Rate: " + str(learning_rate))

    plt.show()


def show_learning_rate_stats(batch_size):
    learning_rates = [0.0008, 0.0004, 0.0002, 0.0001, 0.00005]

    for learning_rate in learning_rates:
        path = "log\\" + str(batch_size) + "_" + str(learning_rate) + ".dat"
        if os.path.isfile(path):
            loss = np.fromfile(path)
            loss = np.clip(loss, 0, 3)
            plt.plot(loss, label="%s Learning Rate" % learning_rate)

    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.legend()
    plt.title("Learning Rate: " + str(batch_size))

    plt.show()


show_learning_rate_stats(128)
