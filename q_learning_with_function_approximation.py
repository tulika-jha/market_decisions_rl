import numpy as np
import random


def q_function(state, theta):
    state = np.insert(state, 0, 1)
    return np.dot(state, theta)


def valuation(state):
    # returns total value of all assets and cash possessed currently
    asset_val = state[0] * state[1]
    cash_in_hand = state[3]
    return asset_val + cash_in_hand


def update_state(curr_state, action):
    next_state = curr_state[:]
    curr_balance = curr_state[-1]
    curr_price = curr_state[1]
    if action == 0:
        # buy shares
        num_shares_traded = int(curr_balance / curr_price)
        next_state[0] += num_shares_traded
        next_state[3] -= num_shares_traded * curr_price
    elif action == 1:
        # sell shares
        num_shares_traded = curr_state[0]
        next_state[0] = 0
        next_state[3] += num_shares_traded * curr_price
    else: # hold the position
        num_shares_traded = 0
    return next_state

def update_theta(theta, actions, curr_state, next_state, reward, discount_factor, learning_rate):
    max_q_next_state = max([q_function(update_state(next_state, a), theta) for a in actions])
    q_bellman = reward + discount_factor * max_q_next_state
    q_curr = q_function(curr_state, theta)

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
                                           learning_rate=0.01,
                                           train_test_split=0.9,
                                           ):
    ma_window_size = 5
    initial_balance = 10000
    epsilon = 0.2
    num_epochs = 100

    df["moving_avg"] = df["Close"].rolling(window=5).mean()
    prices = df["Close"].values
    moving_avg = df["moving_avg"].values

    # state is of the form
    # (num of shares, current price of stock, moving average)
    # initialize current state and model parameters 'theta'
    curr_state = [0, None, None, initial_balance]
    num_features = 4
    theta = np.random.rand(num_features + 1)

    # since the first (ma_window_size - 1) moving averages are nan
    prices = prices[ma_window_size - 1:]
    moving_avg = moving_avg[ma_window_size - 1:]

    # split prices and moving averages according to train test split
    prices, test_prices = prices[:int(train_test_split * len(prices))], prices[int(train_test_split * len(prices)):]
    moving_avg, test_moving_avg = moving_avg[:int(train_test_split * len(moving_avg))], \
        moving_avg[int(train_test_split * len(moving_avg)):]

    t = 0
    for epoch in range(num_epochs):
        print("epoch ", epoch)
        for i in range(len(prices)):
            t += 1
            # take action according to epsilon-greedy
            curr_state[1] = prices[i]
            curr_state[2] = moving_avg[i]

            p = np.random.rand()
            if p < epsilon:
                action = random.choice(range(len(actions)))
            else:
                # take the action that maximizes q-value
                q_vals = np.array([q_function(update_state(curr_state, a), theta) for a in range(len(actions))])
                #print(q_vals)
                action = np.random.choice(np.argwhere(q_vals == np.max(q_vals)).flatten())

            # update next state based on chosen action
            # updates the number of shares and balance only,
            # not the prices, which are updated in the next loop
            next_state = update_state(curr_state, action)

            # reward = V_t - initial investment
            reward = valuation(next_state) - initial_balance #valuation(curr_state)  # - transaction_cost(num_shares_traded, )
            # update model parameters theta
            # learning rate decay
            theta = update_theta(theta, actions, curr_state, next_state, reward, discount_factor, learning_rate * (0.99**t))

            #print(">>>>>>>>>>>>>>>>")
            #print(curr_state, "action = {}".format(action), "reward = {}".format(reward), next_state)

            # go to next state
            curr_state = next_state.copy()

        #print(reward, curr_state)
        #print(valuation(curr_state))

    # We now have an estimate of theta, the value function approximation parameter
    # Generate a policy using this estimate q-value

    total_reward = 0
    initial_balance = 10000
    curr_state = [0, None, None, initial_balance]

    for i in range(len(test_prices)):
        curr_state[1] = test_prices[i]
        curr_state[2] = test_moving_avg[i]

        p = np.random.rand()
        if p < epsilon:
            action = random.choice(range(len(actions)))
        else:
            # take the action that maximizes the q-value of the next state
            q_vals = np.array([q_function(update_state(curr_state, a), theta) for a in range(len(actions))])
            # print(q_vals)
            action = np.random.choice(np.argwhere(q_vals == np.max(q_vals)).flatten())

        # updates the number of shares and balance only,
        # not the prices, which are updated in the next loop
        next_state = update_state(curr_state, action)
        reward = valuation(next_state) - initial_balance #valuation(curr_state) #- transaction_cost()

        total_reward += reward
        print("..................")
        print(curr_state, "action = {}".format(action), "reward = {}".format(reward), next_state)

        #print(action, "balance = {}".format(next_state[-1]))

        curr_state = next_state


    print("total reward accumulated during evaluation: {}".format(total_reward))
    print("final valuation: {}, initial valuation: {}".format(valuation(curr_state), initial_balance))

    return reward





