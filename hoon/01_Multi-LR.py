import tensorflow as tf

#data
x1_data = [1,0,3,0,5]
x2_data = [0,2,0,4,0]
y_data = [1,2,3,4,5]

W1 = tf.Variable(tf.random_uniform([1],-1.0,1.0))
W2 = tf.Variable(tf.random_uniform([1],-1.0,1.0))

b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

#hypothesis
hypothesis = W1 * x1_data + W2 * x2_data + b

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
        print ("Step: %d, Cost = %.04f, 기울기 W1 = %.4f, 기울기 W2 = %.4f, y절편 b = %.4f" %
               (step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b)))