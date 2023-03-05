import pandas as pd
import numpy as np


def simulate_data_from_series(df, num_states_price=100):
    # features to use as states
    # price (discretized), moving average (discretized)
    ma_window_size = 5
    df["moving_avg"] = df["price"].rolling(window=5).mean()
    prices = df["price"].values
    moving_avg = df["moving_avg"].values

    # since the first (ma_window_size - 1) moving averages are nan
    prices = prices[ma_window_size-1:]
    moving_avg = moving_avg[ma_window_size-1:]

    def states_mapping(p, ma):
        price_mapping = min(np.floor(((p - prices.min())/(prices.max() - prices.min())) * num_states_price), num_states_price - 1)
        ma_mapping = min(np.floor(((ma - moving_avg.min())/(moving_avg.max() - moving_avg.min())) * num_states_price), num_states_price - 1)
        return price_mapping * num_states_price + ma_mapping

    def action_mapping(a):
        action_space = {
            "buy": 0,
            "sell": 1,
            "hold": 2,
        }
        return action_space.get(a)

    balance = 5
    holding = 0 # holding = 1 when you have bought a share and haven't sold it yet
    holding_price = None
    actions = ["buy","sell","hold"]

    markov_chain = []
    state_curr = states_mapping(prices[0], moving_avg[0])

    for i in range(len(prices) - 1):
        state_next = states_mapping(prices[i+1], moving_avg[i+1])
        action = "hold"
        reward = 0
        if holding:
            prob_sell = np.random.rand()
            if prob_sell < 0.5:
                action = "sell"
                reward = prices[i] - holding_price - 0.01 * prices[i]
                balance += reward
                holding = 0
        else:
            if prices[i] < balance:
                prob_buy = np.random.rand()
                if prob_buy < 0.5:
                    action = "buy"
                    holding = 1
                    holding_price = prices[i]
                    balance -= (prices[i] + 0.01 * prices[i])
        action = action_mapping(action)
        markov_chain.append([int(state_curr), int(action), reward, int(state_next)])
        state_curr = state_next

    return markov_chain







