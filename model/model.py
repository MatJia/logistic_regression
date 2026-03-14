import generator
import pandas as pd
import numpy as np
weight_temp = generator.Temperature #生成测试集时降温，则最后结果权重也会降温，在这里引入升温系数

def ini_sets() -> None:
    generator.generate_sets()

def sigma(x: float) -> float:
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    ini_sets()
    train_file = pd.read_csv("../data/train_data.csv")
    test_file = pd.read_csv("../data/test_data.csv")
    X = train_file.iloc[:, :-1].to_numpy()
    y = train_file.iloc[:, -1].to_numpy()
    T_X = test_file.iloc[:, :-1].to_numpy()
    T_y = test_file.iloc[:, -1].to_numpy()
