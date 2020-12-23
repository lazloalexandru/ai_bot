import common as cu
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

global t
global high
global ctrl_held
global fig, axes
global df
global title, markers, start_idx, start_marker


def onclick(event):
    global ctrl_held
    global t
    global high
    global df, start_marker
    global title, markers, start_idx, end_idx

    x = int(round(event.xdata))
    if event.button == 1:
        if ctrl_held is not None and ctrl_held:
            print("Start: ", t[x], x)
            start_idx = x
            start_marker = [float('nan')] * len(t)
            start_marker[x] = high[x] * 1.005
        else:
            if start_idx is not None:
                print("End: ", t[x], x)
                start_marker = [float('nan')] * len(t)

                end_idx = x

                if end_idx < start_idx:
                    start_idx, end_idx = end_idx, start_idx

                for i in range(start_idx, end_idx+1):
                    markers[i] = high[i] * 1.005

                start_idx = None
    elif event.button == 3:
        if ctrl_held is not None and ctrl_held:
            markers = [float('nan')] * len(t)
        else:
            print("DEL")
            start_idx = None
            start_marker = [float('nan')] * len(t)


    plot_marker()
    fig.canvas.draw()


def plot_marker():
    global fig, axes, df, title, markers

    axes[0].clear()
    adp = []
    adp.append(mpf.make_addplot(markers, scatter=True, ax=axes[0], markersize=10, marker=r'$\Downarrow$', color='green'))
    adp.append(mpf.make_addplot(start_marker, scatter=True, ax=axes[0], markersize=10, marker=r'$\Downarrow$', color='red'))

    # fig, axes = mpf.plot(df, type='candle', title="zzz", ax=axes[0], volume=axes[3], tight_layout=True, returnfig=True, addplot=adp)
    mpf.plot(df, type='candle', ax=axes[0], addplot=adp, tight_layout=True)


def on_key_press(event):
    global ctrl_held, start_idx, start_marker
    if event.key == 'control':
        ctrl_held = True

def on_key_release(event):
    global ctrl_held
    if event.key == 'control':
        ctrl_held = False


def labeling(symbol, date):
    global t
    global high
    global ctrl_held
    global fig, axes
    global df, markers, start_marker
    global title, start_idx, end_idx

    start_idx = None
    end_idx = None
    ctrl_held = False

    title = symbol + " " + date

    p = {
        '__chart_begin_hh': 9,
        '__chart_begin_mm': 30,
        '__chart_end_hh': 15,
        '__chart_end_mm': 59,
    }
    df = cu.get_chart_data_prepared_for_ai(symbol, date, p)
    df = df.set_index(pd.Index(df.Time))

    t = df.Time.to_list()
    high = df.High.to_list()
    markers = [float('nan')] * len(t)
    start_marker = [float('nan')] * len(t)

    print(len(df))

    fig, axes = mpf.plot(df, type='candle', title=title, returnfig=True, tight_layout=True)

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)

    plt.show()

    for i in range(len(markers)):
        if markers[i] > 0:
            print(t[i])


labeling(symbol="AAL", date="2020-06-16")
