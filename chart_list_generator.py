import common
import pandas as pd
import multiprocessing as mp
import numpy as np
import torch

'''
def _get_active_days_for(symbol, params):
    df = common.get_daily_chart_for(symbol)
    df.Time = pd.to_datetime(df.Time, format="%Y-%m-%d")

    relevant_days_list = []
    n = len(df)

    for i in range(1, n):
        if df.loc[i - 1]['Close'] > 0 and df.loc[i]['Low'] > 0:

            gap = (df.loc[i]['Open'] - df.loc[i - 1]['Close']) / df.loc[i-1]['Close']
            gap = round(gap * 100)  # gap up in percents

            range_ = (df.loc[i]['High'] - df.loc[i]['Low']) / df.loc[i]['Low']
            range_ = round(range_ * 100)  # in percents

            #####################################################################
            #  DEBUG

            # xxx = df["Time"][df.index[i]]
            # if xxx == "2019-05-01":
            #     print("Gap: " + str(gap))
            #    print("Range: " + str(range_))
            #    print(df.loc[i, 'Volume'])

            #####################################################################

            if df.loc[i, 'Volume'] * df.loc[i, 'Close'] > params['min_value_traded']:
                if df.loc[i, 'Open'] > params['day_open_above'] and df.loc[i, 'High'] > params['day_high_above']:
                    if params['min_gap_up'] is not None:
                        if gap > params['min_gap_up']:
                            t = df["Time"][df.index[i]].strftime("%Y-%m-%d")
                            print(symbol, t, "    gap:", gap)
                            relevant_days_list.append([gap, t, symbol])

                    if params['min_range'] is not None:
                        if range_ > params['min_range']:
                            t = df["Time"][df.index[i]].strftime("%Y-%m-%d")
                            print(symbol, t, "     <--- range:", range_)
                            relevant_days_list.append([range_, t, symbol])

    print(symbol + " Relevant Days (Gap-Ups + Big movers): " + str(int(len(relevant_days_list))))
    return relevant_days_list


def find_active_days_mp(params, cpu_count=1):
    symbols = common.get_list_of_symbols_with_daily_chart()

    results = []
    if cpu_count > 1:
        print("Num CPUs: ", cpu_count)

        pool = mp.Pool(cpu_count)
        mp_results = [pool.apply_async(_get_active_days_for, args=(symbol, params)) for symbol in symbols]
        pool.close()
        pool.join()

        for res in mp_results:
            movers = res.get(timeout=1)
            if len(movers) > 0:
                for m in movers:
                    results.append(m)

    else:
        print("Single CPU execution")

        for symbol in symbols:
            movers = _get_active_days_for(symbol, params)
            if len(movers) > 0:
                for m in movers:
                    results.append(m)

    return results
'''


def find_all_days(params):
    symbols = common.get_list_of_symbols_with_daily_chart()

    MILLION = 1000000

    results = []
    for symbol in symbols:
        files = common.get_intraday_chart_files_for(symbol)

        for file in files:
            date = file.replace(".csv", "")[-8:]

            df = common.get_intraday_chart_for(symbol, date)
            t_ = df.Time.to_list()
            price = np.array(df.Close.to_list())
            volume = np.array(df.Volume.to_list())

            val = sum(price * volume)

            print(file + date + "   %sM" % int(val / MILLION), end="")

            if val > params['min_value_traded']:
                results.append([date, symbol])
                print("  <----")
            else:
                print("")

    return results


def write_indexed_chart_list_to_file(chart_list, index_list, path):
    df = pd.DataFrame(columns=['date', 'symbol'])

    n = len(index_list)
    for i in range(0, n):

        data = {
            'date': chart_list[index_list[i]][0],
            'symbol': chart_list[index_list[i]][1]
        }

        df = df.append(data, ignore_index=True)

        if i % 50 == 0 and i > 0:
            print(".", end="")
        if (i % 5000 == 0 and i > 0) or i == (n-1):
            print("")

    common.make_dir(common.__data_dir)
    df.to_csv(path)

    print("Saved", len(index_list), " charts at:", path)


def generate_chart_list_files():
    params = get_default_params()
    chart_list = find_all_days(params)

    num_charts = len(chart_list)

    if num_charts > 0:
        train_set_size = int(num_charts * params['split_train_test'])
        __all_test_set_size = num_charts - train_set_size
        dev_test_set_size = int(__all_test_set_size * params['split_dev_test_test'])
        test_set_size = __all_test_set_size - dev_test_set_size

        print("Number Of Charts: ", num_charts)
        print("Training Set Size: ", train_set_size, "(%.1f" % (100*train_set_size/num_charts), "%)")
        print("Dev Test Set Size: ", dev_test_set_size, "(%.1f" % (100*dev_test_set_size/num_charts), "%)")
        print("Test Set Size: ", test_set_size, "(%.1f" % (100 * test_set_size / num_charts), "%)")

        print("Generating Random Split")
        train_idx, dev_test_idx, test_idx = torch.utils.data.random_split(
            range(num_charts),
            [train_set_size, dev_test_set_size, test_set_size],
            generator=torch.Generator().manual_seed(params['seed'])
        )

        write_indexed_chart_list_to_file(chart_list, train_idx, params['output_training_file_path'])
        write_indexed_chart_list_to_file(chart_list, dev_test_idx, params['output_dev_test_file_path'])
        write_indexed_chart_list_to_file(chart_list, test_idx, params['output_test_file_path'])


def _main():
    generate_chart_list_files()


def get_default_params():
    params = {

        ###################### CHART PARAMETERS ##########################

        'min_value_traded': 10000000,
        'day_high_above': 10,
        'day_open_above': 1,
        'min_gap_up': 5,
        'min_gap_up_volume': 1,
        'min_range': None,
        'min_range_volume': None,

        #################### OUTPUT PARAMETERS ###########################

        'split_train_test': 0.95,
        'split_dev_test_test': 0.25,
        'seed': 91,  # random generator initializer
        'output_training_file_path': "data\\training_charts.csv",
        'output_dev_test_file_path': "data\\dev_test_charts.csv",
        'output_test_file_path': "data\\test_charts.csv"
    }

    return params


if __name__ == "__main__":
    _main()
