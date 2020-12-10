import numpy as np
import matplotlib.pyplot as plt
import os

__loss_history_dir = "analytics\\learning_rate\\loss_history_files"


def show_batch_stats(learning_rate):
    batch_sizes = [16, 32, 64, 128, 256, 512, 1024, 4096]

    for batch_size in batch_sizes:
        path = __loss_history_dir + "\\" + str(batch_size) + "_" + str(learning_rate) + ".dat"
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
    learning_rates = [0.0008, 0.0004, 0.0002, 0.0001, 0.00005, 0.00001, 0.000005]

    for learning_rate in learning_rates:
        path = __loss_history_dir + "\\" + str(batch_size) + "_" + str(learning_rate) + ".dat"
        if os.path.isfile(path):
            loss = np.fromfile(path)
            loss = np.clip(loss, 0, 3)
            plt.plot(loss, label="%s Learning Rate" % learning_rate)

    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.legend()
    plt.title("Learning Rate: " + str(batch_size))

    plt.show()


def show_combined_stats(lr_bs_vec):
    for x in lr_bs_vec:
        batch_size = x[1]
        learning_rate = x[0]

        path = __loss_history_dir + "\\" + str(batch_size) + "_" + str(learning_rate) + ".dat"
        if os.path.isfile(path):
            loss = np.fromfile(path)
            loss = np.clip(loss, 0, 3)
            plt.plot(loss, label="%s / %s" % (learning_rate, batch_size))

    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.legend()
    plt.title("Learning Rate: " + str(batch_size))

    plt.show()


def combined_doubling_stats():
    data = [[0.0008, 1024], [0.0004, 512], [0.0002, 256], [0.0001, 128], [0.00005, 64]]

    show_combined_stats(data)


def combined_stats():
    data = [[0.0001, 256], [0.0001, 128], [0.0001, 64]]

    show_combined_stats(data)


def combined_stats_64():
    data = [[0.0001, 64], [0.00001, 64]]

    show_combined_stats(data)


combined_stats()

# show_learning_rate_stats(64)
