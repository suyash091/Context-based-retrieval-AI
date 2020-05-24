import pandas as pd
from metrics import evaluate_recall
from models import random_model, TFIDFmodel

if __name__ == '__main__':
    #Random Model
    test_df = pd.read_csv('/build_dataset/src/test.csv')
    y_random = [random_model(test_df.Context[x], test_df.iloc[x, 1:].values) for x in range(len(test_df))]
    y_test = np.zeros(len(y_random))
    print('Random Model Score:')
    for n in [1, 2, 5, 10]:
        print('Recall @ ({}, 10): {:g}'.format(n, evaluate_recall(y_random, y_test, n)))
    print('******')

    #TfIdf Model
    train = pd.read_csv('/build_dataset/src/train.csv')
    pred = TFIDFmodel()
    pred.train(train)
    y = [pred.predict(test_df.Context[x], test_df.iloc[x, 1:].values) for x in range(len(test_df))]
    print('TfIdf Model Score:')
    for n in [1, 2, 5, 10]:
        print('Recall @ ({}, 10): {:g}'.format(n, evaluate_recall(y, y_test, n)))
