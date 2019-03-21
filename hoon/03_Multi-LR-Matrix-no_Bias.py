import tensorflow as tf
import numpy as np

#data
x_data = [[1,1,1,1,1],
          [0.,2.,0.,4.,0.],
          [1.,0.,3.,0.,5.]] #x_data에 [1,1,1,1,1] 추가
y_data = [1,2,3,4,5]

#Weight
W = tf.Variable(tf.random_uniform([1,3],-1.0,1.0)) #가중치도 1x3로 변경, b삭제

#hypothesis
hypothesis = tf.matmul(W, x_data) # b 삭제
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

#minimize
a = tf.Variable(0.1) #alpha, learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# befor starting, initialize variables
init = tf.global_variables_initializer()

#launch
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(2001):
    sess.run(train)
    if step % 50 == 0:
        print(step, sess.run(cost), sess.run(W))