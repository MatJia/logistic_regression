import numpy as np
import random as rd
#该样本三个特征值分别为平均每天学习时间（0-8h）/上课出勤率（0-1）/模拟考成绩（40-100）
train_scale = 8000
test_scale = 2000
learning_time_influence = 5 #with_mark 0-40
attendance_influence = 40 #with_mark 0-40
simulate_test_influence = 0.2 #with_mark 8-20
mark_line = 60
Temperature = 5#加入温度系数，以此让整体数据更温和

def learning_time_generator() -> float:
    base_time = np.random.randn() + 5
    base_time = 8 if base_time > 8 else 0 if base_time < 0 else base_time
    return base_time

def attendance_rate_generator() -> float:
    base_att = (np.random.randn() + 3) / 6
    base_att = 1 if base_att > 1 else 0 if base_att < 0 else base_att
    return base_att

def simulate_test_generator() -> float:
    base_score = (np.random.randn() + 10) * 8
    base_score = 100 if base_score > 100 else 40 if base_score < 40 else base_score
    return base_score

def generate_sets() -> None:
    with open("../data/training_data.csv", "w") as file:
        file.write("learning_time,attend_rate,test_score,label\n")
        for i in range(train_scale):
            l_mark = learning_time_generator()
            a_mark = attendance_rate_generator()
            s_mark = simulate_test_generator(l_mark)
            final_score = l_mark * learning_time_influence + a_mark * attendance_influence + s_mark * simulate_test_influence
            final_pass_prob = 1 / (1 + np.exp(-(final_score - mark_line) / Temperature))
            final_pass = rd.uniform(0, 1) < final_pass_prob
            file.write(f"{l_mark},{a_mark},{s_mark},{"1" if final_pass else "0"}\n")

    with open("../data/test_data.csv", "w") as file:
        file.write("learning_time,attend_rate,test_score,label\n")
        for i in range(test_scale):
            l_mark = learning_time_generator()
            a_mark = attendance_rate_generator()
            s_mark = simulate_test_generator()
            final_score = l_mark * learning_time_influence + a_mark * attendance_influence + s_mark * simulate_test_influence
            final_pass_prob = 1 / (1 + np.exp(-(final_score - mark_line) / Temperature))
            final_pass = rd.uniform(0, 1) < final_pass_prob
            file.write(f"{l_mark},{a_mark},{s_mark},{"1" if final_pass else "0"}\n")

if __name__ == "__main__":
    generate_sets()