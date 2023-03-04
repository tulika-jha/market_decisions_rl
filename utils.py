import pandas as pd




def simulate_data_from_series(df, num_states=20):
    # features to use as states
    # price (discretized), moving average (discretized)
    prices = df["price"].values
    df["price_states"] = (prices - prices.min() * num_states / (prices.max() - prices.min())).floor()
    df["moving_avg"] = df["prices"].rolling(window=5).mean()
    df["moving_avg_states"] = (df["moving_avg"].values() - prices.min() * num_states / (prices.max() - prices.min())).floor()
    moving_avg = df["moving_avg"].values()

    balance = 5
    # sold = 1 when you have bought a share
    sold = 0
    actions = ["buy","sell","hold"]

    markov_chain = []
    for i in range(len(prices)):
        prob = np.random.rand()
        action = "hold"
        if prices[i] < balance:
            if prob < 0.5:
                action = "buy"




