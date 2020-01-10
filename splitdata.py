import csv
import pandas as pd
import pickle
import numpy as np
from A1 import A1_data
from A1 import A1_test_data
from A2 import A2_data
from A2 import A2_test_data
from B1 import B1_data
from B1 import B1_test_data
from B2 import B2_data
from B2 import B2_test_data
import keras

def get_data():
    X, y = A1_data.extract_features_labels()
    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:3000] ; tr_Y = Y[:3000]
    te_X = X[3000:] ; te_Y = Y[3000:]

    return tr_X, tr_Y, te_X, te_Y

def get_data_2():
    X, y = A2_data.extract_features_labels()
    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:3000] ; tr_Y = Y[:3000]
    te_X = X[3000:] ; te_Y = Y[3000:]

    return tr_X, tr_Y, te_X, te_Y

def get_test_data():
    X, y = A1_test_data.extract_features_labels()
    Y = np.array([y, -(y - 1)]).T

    return X, Y

def get_test_data_2():
    X, y = A2_test_data.extract_features_labels()
    Y = np.array([y, -(y - 1)]).T

    return X, Y

def get_data_A1():
    X, y = A1_test_data.extract_features_labels()

    # save the face data and label in pickle file
    pickle_out = open('./Datasets/A1_X.pickle', 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open('./Datasets/A1_y.pickle', 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()

    # read teh data from pickle file
    # with open('./test_data/X.pickle', 'rb') as file:
    #    try:
    #        X = pickle.load(file)
    #    except EOFError:
    #        print('X Error')
    # with open('./test_data/y.pickle', 'rb') as file:
    #    try:
    #        y = pickle.load(file)
    #    except EOFError:
    #        print('Y Error')

    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:3000];
    tr_Y = Y[:3000]
    te_X = X[3000:];
    te_Y = Y[3000:]
    val_X = tr_X[:300];
    val_Y = tr_Y[:300]

    return tr_X, tr_Y, te_X, te_Y, val_X, val_Y


def get_test_data_A1():
    X, y = A1_test_data.extract_features_labels()

    # save the face data and label in pickle file
    pickle_out = open('./Datasets/A1_test_X.pickle', 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open('./Datasets/A1_test_y.pickle', 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()

    # read teh data from pickle file
    # with open('./test_data/X.pickle', 'rb') as file:
    #    try:
    #        X = pickle.load(file)
    #    except EOFError:
    #        print('X Error')
    # with open('./test_data/y.pickle', 'rb') as file:
    #    try:
    #        y = pickle.load(file)
    #    except EOFError:
    #        print('Y Error')

    Y = np.array([y, -(y - 1)]).T

    return X, Y

def get_data_A2():
    #X, y = A2_data.extract_features_labels()

    # save the face data and label in pickle file
    #pickle_out = open('./Datasets/A2_X.pickle', 'wb')
    #pickle.dump(X, pickle_out)
    #pickle_out.close()

    #pickle_out = open('./Datasets/A2_y.pickle', 'wb')
    #pickle.dump(y, pickle_out)
    #pickle_out.close()

    # read teh data from pickle file
    with open('./Datasets/A2_X.pickle', 'rb') as file:
       try:
           X = pickle.load(file)
       except EOFError:
           print('X Error')
    with open('./Datasets/A2_y.pickle', 'rb') as file:
       try:
           y = pickle.load(file)
       except EOFError:
           print('Y Error')

    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:3000];
    tr_Y = Y[:3000]
    te_X = X[3000:];
    te_Y = Y[3000:]
    val_X = tr_X[:300];
    val_Y = tr_Y[:300]

    return tr_X, tr_Y, te_X, te_Y, val_X, val_Y

def get_test_data_A2():
    X, y = A2_test_data.extract_features_labels()

    # save the face data and label in pickle file
    pickle_out = open('./Datasets/A2_test_X.pickle', 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open('./Datasets/A2_test_y.pickle', 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()

    # read teh data from pickle file
    #with open('./Datasets/A2_X.pickle', 'rb') as file:
    #   try:
    #       X = pickle.load(file)
    #   except EOFError:
    #       print('X Error')
    #with open('./Datasets/A2_y.pickle', 'rb') as file:
    #   try:
    #       y = pickle.load(file)
    #   except EOFError:
    #       print('Y Error')

    Y = np.array([y, -(y - 1)]).T

    return X, Y

def get_data_B1():
    num_classes = 5

    #X, y = B1_data.extract_features_labels()
    # save the face data and label in pickle file
    #pickle_out = open('./Datasets/B1_X.pickle', 'wb')
    #pickle.dump(X, pickle_out)
    #pickle_out.close()

    #pickle_out = open('./Datasets/B1_y.pickle', 'wb')
    #pickle.dump(y, pickle_out)
    #pickle_out.close()

    # read teh data from pickle file
    with open('./Datasets/B1_X.pickle', 'rb') as file:
        try:
            X = pickle.load(file)
        except EOFError:
            print('X Error')
    with open('./Datasets/B1_y.pickle', 'rb') as file:
        try:
            y = pickle.load(file)
        except EOFError:
            print('Y Error')

    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:5600];
    tr_Y = Y[:5600]
    te_X = X[5600:];
    te_Y = Y[5600:]
    val_X = tr_X[:600];
    val_Y = tr_Y[:600]

    tr_Y = keras.utils.to_categorical(tr_Y[:, 0], num_classes)
    te_Y = keras.utils.to_categorical(te_Y[:, 0], num_classes)
    val_Y = keras.utils.to_categorical(val_Y[:, 0], num_classes)

    return tr_X, tr_Y, te_X, te_Y, val_X, val_Y

def get_test_data_B1():
    num_classes = 5
    # X, y = A1_data.extract_features_labels()
    X, y = B1_test_data.extract_features_labels()
    # save the face data and label in pickle file
    pickle_out = open('./Datasets/B1_test_X.pickle', 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open('./Datasets/B1_test_y.pickle', 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()

    # read teh data from pickle file
    #with open('./Datasets/B2_X.pickle', 'rb') as file:
    #    try:
    #        X = pickle.load(file)
    #    except EOFError:
    #        print('X Error')
    #with open('./Datasets/B2_y.pickle', 'rb') as file:
    #    try:
    #        y = pickle.load(file)
    #    except EOFError:
    #        print('Y Error')

    Y = np.array([y, -(y - 1)]).T


    add_Y = keras.utils.to_categorical(Y[:, 0], num_classes)

    return X, add_Y

def get_data_B2():
    num_classes = 5

    X, y = B2_data.extract_features_labels()
    # save the face data and label in pickle file
    #pickle_out = open('./Datasets/B2_X.pickle', 'wb')
    #pickle.dump(X, pickle_out)
    #pickle_out.close()

    #pickle_out = open('./Datasets/B2_y.pickle', 'wb')
    #pickle.dump(y, pickle_out)
    #pickle_out.close()

    # read teh data from pickle file
    #with open('./Datasets/B2_X.pickle', 'rb') as file:
    #    try:
    #        X = pickle.load(file)
    #    except EOFError:
    #        print('X Error')
    #with open('./Datasets/B2_y.pickle', 'rb') as file:
    #    try:
    #        y = pickle.load(file)
    #    except EOFError:
    #        print('Y Error')

    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:7000];
    tr_Y = Y[:7000]
    te_X = X[7000:];
    te_Y = Y[7000:]
    val_X = tr_X[:700];
    val_Y = tr_Y[:700]

    tr_Y = keras.utils.to_categorical(tr_Y[:, 0], num_classes)
    te_Y = keras.utils.to_categorical(te_Y[:, 0], num_classes)
    val_Y = keras.utils.to_categorical(val_Y[:, 0], num_classes)

    return tr_X, tr_Y, te_X, te_Y, val_X, val_Y

def get_test_data_B2():
    num_classes = 5

    X, y = B2_test_data.extract_features_labels()
    # save the face data and label in pickle file
    pickle_out = open('./Datasets/B2_test_X.pickle', 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open('./Datasets/B2_test_y.pickle', 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()

    # read teh data from pickle file
    #with open('./Datasets/B2_X.pickle', 'rb') as file:
    #    try:
    #        X = pickle.load(file)
    #    except EOFError:
    #        print('X Error')
    #with open('./Datasets/B2_y.pickle', 'rb') as file:
    #    try:
    #        y = pickle.load(file)
    #    except EOFError:
    #        print('Y Error')

    Y = np.array([y, -(y - 1)]).T

    add_Y = keras.utils.to_categorical(Y[:, 0], num_classes)

    return X, add_Y