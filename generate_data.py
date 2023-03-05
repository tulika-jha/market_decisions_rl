import pandas as pd
import random
import os


def random_timeseries(initial_value, volatility, num_time_steps):
    ts = [initial_value]
    for _ in range(num_time_steps - 1):
        ts.append(ts[-1] + initial_value * random.gauss(0, 1) * volatility)
    return ts


def save_random_data():
    num_steps = 100
    prices = random_timeseries(5, 0.15, num_steps)
    time_steps = range(num_steps)

    df = pd.DataFrame({
        "time": time_steps,
        "price": prices,
    })

    df.to_csv("test_data.csv", index=False)


def save_binance_data():
    # Loop through all files in AllData to do this process for everything:
    concatenated = pd.DataFrame(
        columns=["Kline open time", "Open price", "High price", "Low price", "Close price", "Volume", "Kline Close time",
                 "Quote asset volume", "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume",
                 "Unused field, ignore"])

    i = 0
    for filename in os.listdir("Data"):
        f = os.path.join("Data", filename)
        data = pd.read_csv(f, names=["Kline open time", "Open price", "High price", "Low price", "Close price", "Volume",
                                     "Kline close time", "Quote asset volume", "Number of trades",
                                     "Taker buy base asset volume", "Taker buy quote asset volume", "Unused field, ignore"])
        data["Date"] = filename[11:-4]

        concatenated = pd.concat([concatenated, data], ignore_index=True)
        print(i)
        i += 1

    first = concatenated.loc[0]["Kline open time"]
    concatenated["Minutes Passed"] = (concatenated["Kline open time"] - first)/60000
    concatenated["time"] = concatenated["Minutes Passed"]
    concatenated["price"] = concatenated["Close price"]
    concatenated.sort_values(by="time", inplace=True)
    concatenated.to_csv("real_data.csv", index=False)


save_binance_data()

