import generator
import pandas as pd
import random as rnd
import numpy as np
import math

learning_rate = 0.001
feature_scaling = [] #统一scaling到-1 - 1
weight_temp = generator.Temperature #生成测试集时降温，则最后结果权重也会降温，在这里引入升温系数
num_of_weight = 3 #当前已有权重数量，构造w矩阵
epoch = 100000 #Maximum train round

def ini_sets() -> None:
    generator.generate_sets()

def sigma(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def rate(x: float):
    pass

def bce_loss(p: float, y: float):
    return -(y*math.log(p,math.e) + (1-y)*math.log(1-p, math.e))

def generate_ini_weight(n: int) -> list:
    lst = list()
    for i in range(n):
        q = rnd.randint(-5,5)
        lst.append(q)
    return lst

def f_scaling(array: np.array) -> list:
    temp_list = list()
    for i in array[0]:
        #print("primary ", i)
        cur_rate = 1
        while i > 1 or i < -1:
            i /= 10
            cur_rate *= 10
            #print(i)
        temp_list.append(cur_rate)
    #print(temp_list)
    return temp_list

def do_scaling(weight: np.array, scale_rate: list) -> None:
    for i in range(len(weight)):
        weight[i] /= scale_rate

if __name__ == "__main__":
    ini_sets()
    train_file = pd.read_csv("../data/training_data.csv")
    test_file = pd.read_csv("../data/test_data.csv")
    X = train_file.iloc[:, :-1].to_numpy()
    y = train_file.iloc[:, -1].to_numpy()
    T_X = test_file.iloc[:, :-1].to_numpy()
    T_y = test_file.iloc[:, -1].to_numpy()

    feature_scaling = f_scaling(X)
    for j in range(len(feature_scaling)):
        X[:,j] /= feature_scaling[j]
        print(X[:,j])

    cur_w = np.array(generate_ini_weight(num_of_weight))
    cur_b = rnd.randint(-5,5)

    for i in range(epoch):
        #print(f"current train round: {i}")
        #训练集全样本训练
        total_error = 0
        total_loss = 0
        for train_data, lable in zip(X, y):
            #print(train_data, lable)
            #z = x矩阵与w矩阵点积+b
            z = train_data @ cur_w + cur_b
