import json
import os
import errno
import argparse

import pandas as pd
import numpy as np

from preprocess import preprocess
from model import save_model, restore_model, create_model
from train import val_split, fit_model, plot_model_history
from test import test


np.random.seed(0)


def train_main():
    with open('config.json') as f:
        args = json.load(f)

    # make save folder
    try:
        print('Creating checkpoint folder...')
        os.makedirs(args['save_folder'])
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

    # read and preprocess data
    train_data = pd.read_csv(args['train_path'])
    preprocessed_train = preprocess(train_data, args)

    if args['train_model_weights']: # resume training
        model = restore_model(args['save_folder'], args['train_model_weights'])
    else:
        model = create_model(args)
        save_model(model, args['save_folder'])


    # split data for CV
    train_set, val_set = val_split(*preprocessed_train[0], preprocessed_train[1])

    model, history = fit_model(model, train_set, val_set, args)
    plot_model_history(history, args['save_folder'])


def test_main():
    with open('config.json') as f:
        args = json.load(f)

    # read and preprocess data
    test_data = pd.read_csv(args['test_path'])
    test_data.drop('id', axis=1, inplace=True)
    preprocessed_test = preprocess(test_data, args, False)

    # load checkpoint
    model = restore_model(args['save_folder'], args['test_model_weights'])
    model.compile(loss='mse', optimizer='adam')  # only for evaluation

    test(model, preprocessed_test, args, True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNN tester')
    parser.add_argument('--test', action='store_true', default=False)

    args = parser.parse_args()

    if args.test:
        print('Testing...')
        test_main()
    else:
        print('Training...')
        train_main()
