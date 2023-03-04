import numpy as np


def q_learning(data, dataset_config, learning_rate):
    # implements q_learning, uses (s, a, r, s') tuples from data
    num_states = dataset_config['state_space_size']
    num_actions = dataset_config['action_space_size']
    discount_factor = dataset_config['discount_factor']

    # initialize Q-values
    q_table = np.zeros((num_states, num_actions))

    for s, a, r, sp in data:
        # q(s, a) <- q(s, a) + learning_rate * (r + discount_factor * (max a' Q(s', a') - Q(s, a)))
        max_a_q_s_a = np.max(q_table[sp, :])
        q_table[s, a] = q_table[s, a] + learning_rate * (r + discount_factor * max_a_q_s_a - q_table[s, a])

    policy = [np.argmax(q_table[s, :]) + 1 for s in range(num_states)]
    return policy