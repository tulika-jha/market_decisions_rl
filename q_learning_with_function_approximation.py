import numpy as np
import random


def q_function(state, theta):
    state = np.insert(state, 0, 1)
    return np.dot(state, theta)


def valuation(state):
    # returns total value of all assets and cash possessed currently
    asset_val = state[0] * state[1]
    cash_in_hand = state[-1]
    return asset_val + cash_in_hand


def update_state(curr_state, action):
    next_state = curr_state[:]
    curr_balance = curr_state[-1]
    curr_price = curr_state[1]
    if action == 0:
        # buy shares
        num_shares_traded = int(curr_balance / curr_price)
        next_state[0] += num_shares_traded
        next_state[-1] -= num_shares_traded * curr_price
    elif action == 1:
        # sell shares
        num_shares_traded = curr_state[0]
        next_state[0] = 0
        next_state[-1] += num_shares_traded * curr_price
    else: # hold the position
        pass
    return next_state


def update_theta(theta, action, actions, curr_state, next_state, reward, discount_factor, learning_rate):
    max_q_next_state = max([q_function(update_state(next_state, a), theta) for a in actions])
    q_bellman = reward + discount_factor * max_q_next_state
    # q(s,a) of current state upon taking action a is q(s')
    q_curr = q_function(update_state(curr_state, action), theta)

    q_grad = np.array(curr_state)
    q_grad = np.insert(q_grad, 0, 1)
    q_grad = q_grad / max(np.linalg.norm(q_grad), 1)

    theta = theta + learning_rate * (q_bellman - q_curr) * q_grad
    return theta


def q_learning_with_function_approximation(df,
                                           actions={
                                               "buy":0,
                                               "sell":1,
                                               "hold":2
                                           },
                                           discount_factor=0.98,
                                           learning_rate=0.001,
                                           train_test_split=0.8,
                                           sharpe_multiplier=252,
                                           ma_window_size=20,
                                           initial_balance=10000,
                                           epsilon=0.2,
                                           num_epochs=100,
                                           num_features=6,
                                           ):

    df["moving_avg"] = df["Close"].rolling(window=ma_window_size).mean()
    prices = df["Close"].values
    moving_avg = df["moving_avg"].values
    ma_daily = df["Close"].rolling(window=1).mean().values
    ma_weekly = df["Close"].rolling(window=5).mean().values
    ma_monthly = df["Close"].rolling(window=21).mean().values
    volume = df["Volume"].values

    # since the first (ma_window_size - 1) moving averages are nan
    prices = prices[ma_window_size - 1:]
    #moving_avg = moving_avg[ma_window_size - 1:]
    ma_daily = ma_daily[ma_window_size - 1:]
    ma_weekly = ma_weekly[ma_window_size - 1:]
    ma_monthly = ma_monthly[ma_window_size - 1:]
    volume = volume[ma_window_size - 1:]

    # split prices and moving averages according to train-test split
    prices, test_prices = prices[:int(train_test_split * len(prices))], prices[int(train_test_split * len(prices)):]
    moving_avg, test_moving_avg = moving_avg[:int(train_test_split * len(moving_avg))], moving_avg[int(train_test_split * len(moving_avg)):]
    volume, test_volume = volume[:int(train_test_split * len(volume))], volume[int(train_test_split * len(volume)):]
    ma_daily, test_ma_daily = ma_daily[:int(train_test_split * len(ma_daily))], ma_daily[int(train_test_split * len(ma_daily)):]
    ma_weekly, test_ma_weekly = ma_weekly[:int(train_test_split * len(ma_weekly))], ma_weekly[int(train_test_split * len(ma_weekly)):]
    ma_monthly, test_ma_monthly = ma_monthly[:int(train_test_split * len(ma_monthly))], ma_monthly[int(train_test_split * len(ma_monthly)):]

    theta = np.random.normal(0, 1, num_features + 1)

    t = 0
    for epoch in range(num_epochs):

        # state = (num_shares, curr_price, daily price change, weekly price change, monthly price change, balance)
        # curr_state = [0, prices[0], moving_avg[0], initial_balance]
        # curr_state = [0, prices[0], initial_balance]
        curr_state = [0, prices[0], ma_daily[0], ma_weekly[0], ma_monthly[0], initial_balance]

        for i in range(1, len(prices)):
            t += 1

            # take action according to epsilon-greedy
            p = random.uniform(0, 1)
            if p < epsilon:
                action = random.choice(range(len(actions)))
            else:
                # take the action with greatest Q(s, a) value
                q_vals = np.array([q_function(update_state(curr_state, a), theta) for a in range(len(actions))])
                action = np.random.choice(np.argwhere(q_vals == np.max(q_vals)).flatten())

            # update next state based on chosen action
            # updates the number of shares and balance only,
            next_state = update_state(curr_state, action)

            next_state[1] = prices[i]
            #next_state[2] = moving_avg[i]
            next_state[2] = ma_daily[i]
            next_state[3] = ma_weekly[i]
            next_state[4] = ma_monthly[i]

            # reward = V_(t) - V_(t-1)
            reward = valuation(next_state) - valuation(curr_state)
            # update model parameters theta
            # learning rate decay
            theta = update_theta(theta, action, actions, curr_state, next_state, reward, discount_factor, learning_rate/(epoch+1)) # * (0.99**t))
            #print(curr_state, "action = {}".format(action), "reward = {}".format(reward), next_state)

            # go to the next state
            curr_state = next_state[:]

    # We now have an estimate of theta, the value function approximation parameter
    # Generate a policy using this estimate q-value

    total_reward = 0
    #curr_state = [0, test_prices[0], test_moving_avg[0], initial_balance]
    #curr_state = [0, test_prices[0], initial_balance]
    curr_state = [0, test_prices[0], test_ma_daily[0], test_ma_weekly[0], test_ma_monthly[0],
                  initial_balance]

    valuations_diff = []

    for i in range(1, len(test_prices)):

        # choose action according to epsilon-greedy
        p = random.uniform(0, 1)
        if p < 0.2:
            action = random.choice(range(len(actions)))
        else:
            # take the action with maximum corresponding q-value
            q_vals = np.array([q_function(update_state(curr_state, a), theta) for a in range(len(actions))])
            action = np.random.choice(np.argwhere(q_vals == np.max(q_vals)).flatten())

        # updates the number of shares and balance only
        next_state = update_state(curr_state, action)

        next_state[1] = test_prices[i]
        # next_state[2] = moving_avg[i]
        next_state[2] = test_ma_daily[i]
        next_state[3] = test_ma_weekly[i]
        next_state[4] = test_ma_monthly[i]

        reward = valuation(next_state) - valuation(curr_state) #initial_balance #valuation(curr_state) #- transaction_cost()
        valuations_diff.append(reward)
        total_reward += reward

        curr_state = next_state[:]

    print("total reward accumulated during evaluation: {}".format(total_reward))
    print("final valuation: {}, initial valuation: {}".format(valuation(curr_state), initial_balance))
    valuations_diff = np.array(valuations_diff)
    sharpe_ratio = valuations_diff.mean() / valuations_diff.std()
    annual_sharpe = sharpe_ratio * np.sqrt(sharpe_multiplier)
    print("sharpe ratio = {}, annualized sharpe ratio = {}".format(sharpe_ratio, annual_sharpe))

    return sharpe_ratio, annual_sharpe





