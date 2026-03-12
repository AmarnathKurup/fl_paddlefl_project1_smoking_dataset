import pandas as pd
import numpy as np


def load_train():

    x = pd.read_csv("data/x_train.csv")
    y = pd.read_csv("data/y_train.csv")

    # remove ID if exists
    x = x.drop(columns=["ID"], errors="ignore")
    y = y.drop(columns=["ID"], errors="ignore")

    # convert categorical columns
    x = x.replace({"Y":1,"N":0,"M":1,"F":0})

    x = x.values.astype("float32")
    y = y.values.astype("int64")

    return x,y


def load_test():

    x = pd.read_csv("data/x_test.csv")
    y = pd.read_csv("data/y_test.csv")

    x = x.drop(columns=["ID"], errors="ignore")
    y = y.drop(columns=["ID"], errors="ignore")

    x = x.replace({"Y":1,"N":0,"M":1,"F":0})

    x = x.values.astype("float32")
    y = y.values.astype("int64")

    return x,y


def trainer_reader(trainer_id):

    x,y = load_train()

    split = len(x)//2

    if trainer_id == 0:
        x = x[:split]
        y = y[:split]
    else:
        x = x[split:]
        y = y[split:]

    for i in range(len(x)):
        yield x[i],y[i]


def test_reader():

    x,y = load_test()

    for i in range(len(x)):
        yield x[i],y[i]