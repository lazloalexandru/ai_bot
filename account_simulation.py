import os
from termcolor import colored
import pandas as pd
import matplotlib.pyplot as plt


'''
__stops = [-4, -5, -6, -7, -8, -10, -12, -15, -20, -25]
__targets = [5, 7, 8, 9, 10, 12, 13, 15, 18, 20, 25, 30, 35, 40, 45, 50, 60]

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def calc_profits_for_stop_target(ms_pattern_entries_filename, params):
    ns = len(__stops)
    nt = len(__targets)

    df = pd.DataFrame(columns=['stop', 'target', 'profit'])

    for i in range(0, ns):
        for j in range(0, nt):
            params['stop'] = __stops[i]
            params['target'] = __targets[j]
            av = _simulate_pattern_flexible(params, 2, ms_pattern_entries_filename)

            data = {'stop': __stops[i],
                    'target': __targets[j],
                    'profit': int(av[-1]/1000)}
            df = df.append(data, ignore_index=True)

            print("\n===============")
            print("STOP: ", __stops[i], " Target: ", __targets[j])

    result_filename = ms_pattern_entries_filename + ".txt"
    df.to_csv(result_filename)



def show_sim_results_for_stop_target(stop_trarget_profit_filename):
    xxx = np.genfromtxt(stop_trarget_profit_filename, dtype=int, delimiter=' ')
    ns = len(__stops)
    nt = len(__targets)

    df = pd.DataFrame(columns=['stop', 'target', 'profit'])

    for i in range(0, ns):
        for j in range(0, nt):
            data = {'stop': __stops[i],
                    'target': __targets[j],
                    'profit': xxx[i, j]}

            df = df.append(data, ignore_index=True)

    # fig = go.Figure(data=[go.Mesh3d(x=df.stop, y=df.target, z=df.profit)])
    fig = px.scatter_3d(df, x='stop', y='target', z='profit')

    fig.show()
'''


def simulate_pattern(sim_params, risk_percent_of_account_per_trade, trading_file_path):
    account_value = sim_params['account_value']
    account_history = []

    gains = []

    num_stops = 0
    num_timeouts = 0

    if os.path.isfile(trading_file_path):
        df = pd.read_csv(trading_file_path)
        df = df.sort_values(by=['entry_time'])

        account_history = [account_value]

        m = len(df)
        i = 0
        while i < m:
            if df['exit_type'][i] == 'STOP':
                num_stops += 1
            elif df['exit_type'][i] == 'Timeout':
                num_timeouts += 1

            risked_value = risk_percent_of_account_per_trade * account_value / 100
            stop_percents = df['stop'][i]
            min_stop = -5
            if stop_percents > min_stop:
                stop_percents = min_stop

            position_size = -(risked_value * 100 / stop_percents)

            max_position = account_value / 4
            if position_size > max_position:
                position_size = max_position

            if position_size > sim_params['size_limit']:
                position_size = sim_params['size_limit']

            risked_value = -df['stop'][i] * position_size / 100

            gain = 100 * (df['sell_price'][i] - df['entry_price'][i]) / df['entry_price'][i]

            color = "green" if gain > 0 else "red"

            account_value = account_value + position_size * gain / 100
            '''
            print(colored(df['sym'][i] + ' ' + df['entry_time'][i] + ' BUY: ' + "%.2f" % df['entry_price'][i], color=color), end="")
            print(colored(' ---> ' + df['sell_time'][i] + ' SOLD: ' + "%.2f" % df['sell_price'][i], color=color), end="")
            print('   Gain: ' + "%.2f" % gain, end="")
            print('    STOP:', int(df['stop'][i]), '    Position Size: ', int(position_size), ' Risked Value: ', int(risked_value), " Account: ", int(account_value))
            '''
            print(df['sym'][i] + ' ' + df['entry_time'][i] + ' BUY: ' + "%.2f" % df['entry_price'][i], end="")
            print(' ---> ' + df['sell_time'][i] + ' SOLD: ' + "%.2f" % df['sell_price'][i], end="")
            print('   Gain:', colored("%.2f %s" % (gain, '%'), color=color), end="")
            print('    STOP:', int(df['stop'][i]), '    Position Size: ', int(position_size), ' Risked Value: ', int(risked_value), " Account: ", int(account_value))

            gains.append(gain)
            account_history.append(account_value)

            i = i + 1
    else:
        print(colored(trading_file_path + ' not found!', color='red'))

    plus = sum(x > 0 for x in gains)
    splus = sum(x for x in gains if x > 0)
    minus = sum(x < 0 for x in gains)
    sminus = sum(x for x in gains if x < 0)

    num_trades = len(gains)
    success_rate = None if (plus + minus) == 0 else round(100 * (plus / (plus + minus)))
    rr = None if plus == 0 or minus == 0 else -(splus/plus)/(sminus/minus)

    avg_win = None if plus == 0 else splus/plus
    avg_loss = None if minus == 0 else sminus/minus

    stopout_rate = "N/A" if num_trades == 0 else str(round(100 * num_stops / num_trades)) + "%"
    timeout_rate = "N/A" if num_trades == 0 else str(round(100 * num_timeouts / num_trades)) + "%"
    hit_rate = "N/A" if num_trades == 0 else str(round(100 * (num_trades - num_stops - num_timeouts) / num_trades)) + "%"

    if len(account_history) > 0:
        print("")
        print("Nr Trades:", num_trades, "   (", hit_rate, "Sell,", stopout_rate, "Stops,", timeout_rate, "Timeouts )")
        print("Success Rate:", success_rate, "%")
        print("R/R:", "N/A" if rr is None else "%.2f" % rr)
        print("Winners:", plus, " Avg. Win:", "N/A" if avg_win is None else "%.2f" % avg_win + "%")
        print("Losers:", minus, " Avg. Loss:", "N/A" if avg_loss is None else "%.2f" % avg_loss + "%")
        print("")

    stats = {
        'num_trades': num_trades,
        'hit_rate': hit_rate,
        'stopout_rate': stopout_rate,
        'timeout_rate': timeout_rate,
        'success_rate': success_rate,
        'rr': rr,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'account_value': account_value
    }

    return account_history, stats


def simulate_account_performance_for(sim_params, trading_file_path):
    risk1 = 1

    ah1, stats = simulate_pattern(sim_params, risk1, trading_file_path)
    # ah2, stats = simulate_pattern(sim_params, 2, trading_file_path)

    if len(ah1) > 0:  # or len(ah2) > 0:
        plt.plot(ah1, label="%.2f Risk" % risk1)
        # plt.plot(ah2, label="2% Risk")

        plt.ylabel("Account Value [$]")
        plt.xlabel("Trades")
        plt.legend()
        plt.title("Account Performance")

        plt.show()

    return stats
