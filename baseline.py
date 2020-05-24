import pandas as pd
from metrics import evaluate_recall
from models import random_model, TFIDFmodel
import argparse
import numpy as np


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='List the content of a folder')

    # Add the arguments
    my_parser.add_argument('train',
                           metavar='train',
                           type=str,
                           help='the path to train csv',
                           default='/build_dataset/src/train.csv')
    my_parser.add_argument('test',
                           metavar='test',
                           type=str,
                           help='the path to test csv',
                           default='/build_dataset/src/test.csv')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    train_path = args.train
    test_path = args.test

    #Random Model
    test_df = pd.read_csv(test_path)
    y_random = [random_model(test_df.Context[x], test_df.iloc[x, 1:].values) for x in range(len(test_df))]
    y_test = np.zeros(len(y_random))
    print('Random Model Score:')
    for n in [1, 2, 5, 10]:
        print('Recall @ ({}, 10): {:g}'.format(n, evaluate_recall(y_random, y_test, n)))
    print('******')

    #TfIdf Model
    train = pd.read_csv(train_path)
    pred = TFIDFmodel()
    pred.train(train)
    y = [pred.predict(test_df.Context[x], test_df.iloc[x, 1:].values) for x in range(len(test_df))]
    print('TfIdf Model Score:')
    for n in [1, 2, 5, 10]:
        print('Recall @ ({}, 10): {:g}'.format(n, evaluate_recall(y, y_test, n)))
