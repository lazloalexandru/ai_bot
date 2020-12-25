import common as cu
import pandas as pd
import multiprocessing as mp
import torch


def write_indexed_chart_list_to_file(chart_list, index_list, path):
    df = pd.DataFrame(columns=['date', 'symbol'])

    n = len(index_list)
    for i in range(0, n):

        data = {
            'date': chart_list[index_list[i]][1],
            'symbol': chart_list[index_list[i]][2]
        }

        df = df.append(data, ignore_index=True)

        if i % 50 == 0 and i > 0:
            print(".", end="")
        if (i % 5000 == 0 and i > 0) or i == (n-1):
            print("")

    cu.make_dir(cu.__data_dir)
    df.to_csv(path)

    print("Saved", len(index_list), " charts at:", path)


def _get_active_days_for(symbol, params):
    df = cu.get_daily_chart_for(symbol)
    df["avg_vol_20"] = df["Volume"].rolling(window=20).mean()
    df.Time = pd.to_datetime(df.Time, format="%Y-%m-%d")

    relevant_days_list = []
    n = len(df)

    for i in range(1, n):
        if df.loc[i - 1]['Close'] > 0 and df.loc[i]['Low'] > 0:

            gap = (df.loc[i]['Open'] - df.loc[i - 1]['Close']) / df.loc[i-1]['Close']
            gap = round(gap * 100)  # gap up in percents

            range_ = (df.loc[i]['High'] - df.loc[i]['Low']) / df.loc[i]['Low']
            range_ = round(range_ * 100)  # gap up in percents

            #####################################################################
            #  DEBUG
            '''
            xxx = df["Time"][df.index[i]]
            if xxx == "2019-05-01":
                print("Gap: " + str(gap))
                print("Range: " + str(range_))
                print(df.loc[i, 'Volume'])
            '''
            #####################################################################

            save_gap = False
            if params['big_gap_up'] is not None:
                if gap > params['big_gap_up']:
                    if df.loc[i, 'Open'] > params['big_gap_up_open']:
                        if df.loc[i, 'Volume'] * df.loc[i, 'Close'] > params['big_gap_up_value_traded']:
                            save_gap = True

            if params['small_gap_up'] is not None:
                if gap > params['small_gap_up']:
                    if df.loc[i, 'Open'] > params['small_gap_up_open']:
                        if df.loc[i, 'Volume'] > params['small_gap_up_volume']:
                            save_gap = True

            if save_gap:
                date = df["Time"][df.index[i]].strftime("%Y-%m-%d")
                print(symbol, date, "    <-    GAP:", gap, end="")

                if cu.intraday_chart_exists_for(symbol, date):
                    relevant_days_list.append([gap, date, symbol])
                    print("")
                else:
                    print("  *** NO CHART ***")
            else:
                save_range = False
                if params['big_range'] is not None:
                    if range_ > params['big_range']:
                        if df.loc[i, 'High'] > params['big_range_high']:
                            if df.loc[i, 'Volume'] * df.loc[i, 'Close'] > params['big_range_value_traded']:
                                if df.loc[i, 'avg_vol_20'] * params['big_range_r_vol'] < df.loc[i, 'Volume']:
                                    save_range = True

                if params['small_range'] is not None:
                    if range_ > params['small_range']:
                        if df.loc[i, 'High'] > params['small_range_high']:
                            if df.loc[i, 'Volume'] > params['small_range_volume']:
                                if df.loc[i, 'avg_vol_20'] * params['small_range_r_vol'] < df.loc[i, 'Volume']:
                                    save_range = True

                if save_range:
                    date = df["Time"][df.index[i]].strftime("%Y-%m-%d")
                    print(symbol, date, "    <-    RANGE:", range_, end="")

                    if cu.intraday_chart_exists_for(symbol, date):
                        relevant_days_list.append([range_, date, symbol])
                        print("")
                    else:
                        print("  *** NO CHART ***")

    # print(symbol, " Relevant Days (Gap-Ups + Big movers):", len(relevant_days_list), "\n")
    return relevant_days_list


def find_active_days_mp(params, cpu_count=1):
    symbols = cu.get_list_of_symbols_with_daily_chart()

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


def generate_chart_list_files():
    params = get_default_params()
    chart_list = find_active_days_mp(params, 16)

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

        write_indexed_chart_list_to_file(chart_list, range(num_charts), params['all_tradeable_charts_file_path'])
        write_indexed_chart_list_to_file(chart_list, train_idx, params['output_training_file_path'])
        write_indexed_chart_list_to_file(chart_list, dev_test_idx, params['output_dev_test_file_path'])
        write_indexed_chart_list_to_file(chart_list, test_idx, params['output_test_file_path'])


def generate_list_of_available_charts(chart_list_file_path):
    df = pd.read_csv(chart_list_file_path)

    n = len(df)
    print("Input -> Number of daily charts:", n)
    for i in range(n):
        print(df.symbol[i], df.date[i])


def _main():
    generate_chart_list_files()


def get_default_params():
    params = {

        'big_gap_up': 5,
        'big_gap_up_open': 2,
        'big_gap_up_value_traded': 200000000,

        'big_range': 10,
        'big_range_high': 2,
        'big_range_value_traded': 200000000,
        'big_range_r_vol': 2,

        'small_gap_up': 10,
        'small_gap_up_open': 2,
        'small_gap_up_volume': 10000000,

        'small_range': 10,
        'small_range_high': 2,
        'small_range_volume': 10000000,
        'small_range_r_vol': 2,

        #################### OUTPUT PARAMETERS ###########################

        'split_train_test': 0.95,
        'split_dev_test_test': 0.25,
        'seed': 91,  # random generator initializer
        'output_training_file_path': "data\\training_charts.csv",
        'output_dev_test_file_path': "data\\dev_test_charts.csv",
        'output_test_file_path': "data\\test_charts.csv",
        'all_tradeable_charts_file_path': "data\\all_tradeable_charts.csv"
    }

    return params


if __name__ == "__main__":
    _main()
