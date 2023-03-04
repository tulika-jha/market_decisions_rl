import pandas as pd
import random


def random_timeseries(initial_value, volatility, num_time_steps):
    ts = [initial_value]
    for _ in range(num_time_steps - 1):
        ts.append(ts[-1] + initial_value * random.gauss(0, 1) * volatility)
    return ts

num_steps = 100
prices = random_timeseries(1.2, 0.15, num_steps)
time_steps = range(num_steps)


df = pd.DataFrame({
    "time": time_steps,
    "price": prices,
})

df.to_csv("test_data.csv", index=False)
