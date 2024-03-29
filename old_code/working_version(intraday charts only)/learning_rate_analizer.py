import numpy as np
import matplotlib.pyplot as plt
import os

from termcolor import colored

__loss_history_dir = "analytics\\learning_rate\\loss_history_files"
__loss_history_weight_decay_dir = "analytics\\weight_decay\\loss_history_files"


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
    plt.title("Batch Size: " + str(batch_size))

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
    plt.title("Learning Rate & Batch Size")

    plt.show()


def show_combined_stats_weight_decay(lr_bs_wd_vec):
    for x in lr_bs_wd_vec:
        batch_size = x[1]
        learning_rate = x[0]
        weight_decay = x[2]

        path = __loss_history_weight_decay_dir + "\\" + str(batch_size) + "_" + str(learning_rate) + "_" + str(weight_decay) + ".dat"
        if os.path.isfile(path):
            loss = np.fromfile(path)
            loss = np.clip(loss, 0, 3)
            plt.plot(loss, label="%s / %s / %s" % (learning_rate, batch_size, weight_decay))
        else:
            print(colored("File not found:" + path, color='red'))

    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.legend()
    plt.title("Learning Rate & Batch Size & Weight Decay")

    plt.show()


def combined_doubling_stats():
    data = [[0.0008, 1024], [0.0004, 512], [0.0002, 256], [0.0001, 128], [0.00005, 64]]

    show_combined_stats(data)


def combined_stats():
    data = [[0.0001, 256], [0.0001, 128], [0.0001, 64]]

    show_combined_stats(data)


def combined_stats_weight_decay():
    data = [[0.0001, 128, 0], [0.0001, 128, 0.1], [0.0001, 128, 0.01], [0.0001, 128, 0.001]]

    show_combined_stats_weight_decay(data)


def combined_stats_64():
    data = [[0.0001, 64], [0.00001, 64]]

    show_combined_stats(data)


combined_stats_weight_decay()

# show_learning_rate_stats(64)
