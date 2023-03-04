import argparse
import pandas as pd
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
    m_chain_data = simulate_data_from_series(dataset)

    # solve the problem using the specified method
    rl_func = ALGORITHM[method]
    policy = rl_func(m_chain_data)

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
