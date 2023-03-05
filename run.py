import argparse
import pandas as pd
import numpy as np
from q_learning import q_learning
from utils import simulate_data_from_series


ALGORITHM = {
    "ql": q_learning,
}


def run_rl(filename, method):
    # load and preprocess data
    # dataset = load_and_preprocess_data(filename)
    dataset = pd.read_csv(filename)
    # dataset is a pandas DataFrame with the columns (time, price)
    m_chain_data = simulate_data_from_series(dataset, num_states_price=100)

    #print(m_chain_data)
    #print(np.array(m_chain_data).max())

    # solve the problem using the specified method
    rl_func = ALGORITHM[method]
    dataset_config = {
        'state_space_size': 100*100,
        'action_space_size': 3,
        'discount_factor': 0.98,
    }
    policy = rl_func(data=m_chain_data, dataset_config=dataset_config, learning_rate=0.1)
    print(policy)
    # evaluate policy



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename')
    parser.add_argument('--method')
    args = parser.parse_args()
    filename = args.filename
    method = args.method

    run_rl(filename, method)
