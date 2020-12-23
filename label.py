from time import sleep
import common as cu
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
import os


class LabelingTool:
    def __init__(self, chart_list_file_path, review_mode=False):
        if not os.path.isfile(chart_list_file_path):
            raise ValueError("File not found: " + chart_list_file_path)

        self.initialized = False

        self.symbol = None
        self.date = None
        self.t = None
        self.high = None
        self.df = None

        self.ctrl_held = None

        self.fig = None
        self.axes = None

        self.start_idx = None
        self.markers = None
        self.start_marker = None
        self.labeled = None

        self.review_mode = review_mode
        self.chart_list = pd.read_csv(chart_list_file_path)

        if len(self.chart_list) <= 0:
            raise ValueError("No charts in file: " + chart_list_file_path)

        self.chart_idx = 0

        self.select_chart(1)

    def skip_to_next_chart(self, direction):
        n = len(self.chart_list)
        i = self.chart_idx + direction

        if 0 < i < n:
            self.symbol = self.chart_list.iloc[i]["symbol"]
            self.date = self.chart_list.iloc[i]["date"]
            self.chart_idx = i

    def skip_to_next_unlabeled_chart(self, direction):
        symbol = None
        date = None

        n = len(self.chart_list)
        found = False
        i = self.chart_idx + direction

        while 0 <= i < n and not found:
            symbol = self.chart_list.iloc[i]["symbol"]
            date = self.chart_list.iloc[i]["date"]

            path = cu.get_label_file_path(symbol, date)
            if not os.path.isfile(path):
                found = True

            i += direction

        if found:
            self.symbol = symbol
            self.date = date

    def select_chart(self, direction):
        if self.review_mode:
            self.skip_to_next_chart(direction)
            print("1")
        else:
            print("2")
            self.skip_to_next_unlabeled_chart(direction)

        print(self.symbol, self.date)

        ########################################################################
        # Load chart data

        p = {
            '__chart_begin_hh': 9,
            '__chart_begin_mm': 30,
            '__chart_end_hh': 15,
            '__chart_end_mm': 59,
        }
        self.df = cu.get_chart_data_prepared_for_ai(self.symbol, self.date, p)
        self.df = self.df.set_index(pd.Index(self.df.Time))

        self.t = self.df.Time.to_list()
        self.high = self.df.High.to_list()

        ########################################################################
        # Load labels is exist

        self.start_marker = [float('nan')] * len(self.t)

        self.markers = cu.load_labels(self.symbol, self.date)

        if self.markers is None:
            self.labeled = False
            self.markers = [float('nan')] * len(self.t)
        else:
            self.labeled = True

        ########################################################################

        if not self.initialized:
            self.initialized = True
            label_info = "Labeled" if self.labeled else "Unlabeled"
            title = self.symbol + " " + str(self.date) + "  ->  " + label_info
            self.fig, self.axes = mpf.plot(self.df, type='candle', volume=True, title=title, returnfig=True, tight_layout=True)

            self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

            # fm = plt.get_current_fig_manager()
            # fm.full_screen_toggle()

        self.plot_marker()

        plt.show()

    def plot_marker(self):
        self.axes[0].clear()

        adp = []
        adp.append(mpf.make_addplot(self.markers, scatter=True, ax=self.axes[0], markersize=10, marker=r'$\Downarrow$', color='green'))
        adp.append(mpf.make_addplot(self.start_marker, scatter=True, ax=self.axes[0], markersize=10, marker=r'$\Downarrow$', color='red'))

        mpf.plot(self.df, type='candle', ax=self.axes[0], addplot=adp, tight_layout=True)

        label_info = "Labeled" if self.labeled else "Unlabeled"
        title = self.symbol + " " + str(self.date) + "  ->  " + label_info
        self.fig.suptitle(title, y=0.95)
        self.fig.patch.set_facecolor('palegreen' if self.labeled else "white")

        self.fig.canvas.draw()

    def set_ctrl(self, held):
        self.ctrl_held = held

    def ctrl_held(self):
        return self.ctrl_held

    def mouse_click(self, button, x):
        if button == 1:
            if self.start_idx is None:
                print("Start: ", self.t[x], x)
                self.start_idx = x
                self.start_marker = [float('nan')] * len(self.t)
                self.start_marker[x] = self.high[x] + 0.05
            else:
                print("End: ", self.t[x], x)
                self.start_marker = [float('nan')] * len(self.t)

                end_idx = x

                if end_idx < self.start_idx:
                    self.start_idx, end_idx = end_idx, self.start_idx

                for i in range(self.start_idx, end_idx + 1):
                    self.markers[i] = self.high[i] + 0.05

                self.start_idx = None

        elif button == 3:
            if self.ctrl_held is not None and self.ctrl_held:
                self.markers = [float('nan')] * len(self.t)
            else:
                print("DEL")
                self.start_idx = None
                self.start_marker = [float('nan')] * len(self.t)

        self.plot_marker()

    def onclick(self, event):
        if event.xdata is not None:
            self.mouse_click(event.button, int(round(event.xdata)))

    def on_key_press(self, event):
        if event.key == 'control':
            self.set_ctrl(True)

        elif event.key == 'delete':

            path = cu.get_label_file_path(self.symbol, self.date)
            if os.path.isfile(path):
                print("Removed:", path)
                os.remove(path)

            self.labeled = False
            self.markers = [float('nan')] * len(self.t)
            self.plot_marker()

        elif event.key == ' ':

            self.labeled = True
            self.save_labels()
            self.plot_marker()

        elif event.key == 'left':

            self.select_chart(direction=-1)

        elif event.key == 'right':

            self.select_chart(direction=1)

    def on_key_release(self, event):
        if event.key == 'control':
            self.set_ctrl(False)

    def save_labels(self):
        path = cu.get_label_file_path(self.symbol, self.date)
        cu.save_labels(self.markers, path)
        print("Save labels: ", path)


def labeling():
    labeling_tool = LabelingTool("data\\all_tradeable_charts.csv", review_mode=True)

    '''
    for i in range(len(markers)):
        if markers[i] > 0:
            print(t[i])
    '''

labeling()
