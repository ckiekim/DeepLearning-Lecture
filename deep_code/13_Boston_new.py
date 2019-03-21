#-*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv("../dataset/housing.csv", delim_whitespace=True, header=None)
'''
print(df.info())
print(df.head())
'''
dataset = df.values
X = dataset[:, 0:13]
Y = dataset[:, 13]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
#model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, Y_train, epochs=2000, batch_size=10)
print()

# 예측 값과 실제 값의 비교
Y_prediction = model.predict(X_test).flatten()
sum_error_abs = 0
for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    error = prediction/label - 1
    sum_error_abs += np.abs(error)
    print("실제가격: {:.3f}, 예상가격: {:.3f}, {:.2f}%".format(label, prediction, error*100))
print("에러 절대치의 합: {:.4f}".format(sum_error_abs))
