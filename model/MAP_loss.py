import generator
import pandas as pd
import random as rnd
import numpy as np
import math

warm_temperature = generator.Temperature

def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))

def ini_parament(n: int) -> list:
    temp = list()
    for i in range(n):
        temp.append(rnd.uniform(-1,1))
    return temp

def bce(p: float, y: float) -> float:
    return -(y * np.log(p) + (1-y)*np.log(1-p))

if __name__ == "__main__":
    X = pd.read_csv("../data/training_data.csv").iloc[:,:-1].to_numpy()
    y = pd.read_csv("../data/training_data.csv").iloc[:,-1].to_numpy()
    X_t = pd.read_csv("../data/test_data.csv").iloc[:,:-1].to_numpy()
    y_t = pd.read_csv("../data/test_data.csv").iloc[:,-1].to_numpy()

    weight_amount = X.shape[1]

    w = np.array(ini_parament(weight_amount)).reshape(-1,1)
    b = np.array(ini_parament(1))
    print(X.shape)
    epoch_total = 10000
    lr = 0.001
    lmbda = 0.1
    for epoch in range(epoch_total):
        #gradient of w
        error_vector = list()
        loss_vector = list()
        for features, label in zip(X,y):
            p = sigmoid(features @ w + b)
            error_vector.append(p - label)
            loss_vector.append(bce(p, label))
        gradient_w = X.T @ np.array(error_vector) / X.shape[0]
        gradient_w += (lmbda / X.shape[0]) * w
        gradient_b = np.mean(np.array(error_vector))
        #gradient descent
        w -= lr * gradient_w
        b -= lr * gradient_b
        if epoch % 101 == 1:
            print(f"w_gradient:{gradient_w}\nb_gradient:{gradient_b}")
            print(f"loss: {np.mean(np.array(loss_vector))}")
            print(f"final_weight: {w * warm_temperature}\n")