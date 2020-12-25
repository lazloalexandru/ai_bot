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
        self.manually_labeled = None

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
            self.start_idx = None
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
            self.start_idx = None

            self.symbol = symbol
            self.date = date

    def init_markers_from(self, labels):
        self.markers = [float('nan')] * len(self.t)
        n = len(labels)

        for i in range(n):
            if labels[i] == 1:
                self.markers[i] = self.high[i]

    def labels_from_markers(self):
        n = len(self.t)
        labels = np.zeros(n)

        for i in range(n):
            if not np.isnan(self.markers[i]):
                labels[i] = 1

        return labels

    def load_markers(self):
        self.start_idx = None
        self.start_marker = [float('nan')] * len(self.t)
        labels, label_type = cu.get_labels(self.symbol, self.date)

        if labels is None:
            self.manually_labeled = False
            self.markers = [float('nan')] * len(self.t)
        else:
            self.init_markers_from(labels)
            self.manually_labeled = True if label_type == 1 else False

    def clear_markers(self):
        self.start_idx = None
        self.start_marker = [float('nan')] * len(self.t)
        self.markers = [float('nan')] * len(self.t)

    def select_chart(self, direction):
        if self.review_mode:
            self.skip_to_next_chart(direction)
        else:
            self.skip_to_next_unlabeled_chart(direction)

        print("")
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
        # Load labels if exist

        self.start_marker = [float('nan')] * len(self.t)
        self.load_markers()

        ########################################################################

        if not self.initialized:
            self.initialized = True
            label_info = "Labeled" if self.manually_labeled else "Unlabeled"
            title = self.symbol + " " + str(self.date) + "  ->  " + label_info
            self.fig, self.axes = mpf.plot(self.df, type='candle', volume=True, title=title, returnfig=True, tight_layout=True)

            self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
            self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

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

        label_info = "Labeled" if self.manually_labeled else "Unlabeled"
        title = self.symbol + " " + str(self.date) + "  ->  " + label_info
        self.fig.suptitle(title, y=0.95)

        if self.manually_labeled:
            self.fig.patch.set_facecolor("red" if np.isnan(self.markers).all() else 'limegreen')
        else:
            self.fig.patch.set_facecolor("white")

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
                self.start_idx = None
                self.start_marker = [float('nan')] * len(self.t)

        self.plot_marker()

    def onclick(self, event):
        if event.xdata is not None:
            self.mouse_click(event.button, int(round(event.xdata)))

    def on_scroll(self, event):
        # print(event.button)

        if event.button == 'up':
            self.select_chart(direction=-1)
            sleep(0.2)

        elif event.button == 'down':
            self.select_chart(direction=1)
            sleep(0.2)

    def on_key_press(self, event):
        if event.key == 'control':
            self.set_ctrl(True)

        elif event.key == 'delete':

            if self.manually_labeled:
                path = cu.get_manual_label_path_for(self.symbol, self.date)
                if os.path.isfile(path):
                    print("Removed:", path)
                    os.remove(path)

                    self.load_markers()
            else:
                self.clear_markers()

            self.plot_marker()

        elif event.key == 'enter':

            self.manually_labeled = True
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
        path = cu.get_manual_label_path_for(self.symbol, self.date)

        labels = self.labels_from_markers()

        np.save(path, labels)
        print("Saved labels  -> ", path)


def labeling():
    labeling_tool = LabelingTool("data\\all_tradeable_charts.csv", review_mode=True)

    '''
    for i in range(len(markers)):
        if markers[i] > 0:
            print(t[i])
    '''


labeling()
