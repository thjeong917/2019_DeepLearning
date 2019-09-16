import tensorflow as tf
import numpy as np
import random
import math

def result(u, v):
    if u*u < v:
        return 0
    return 1

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.int32, shape=[None])

n=2;

Y_onehot = tf.one_hot(Y, n)

W1_1 = tf.Variable(tf.random_normal([2,2]), name='W1_1')
b1_1 = tf.Variable(tf.random_normal([2]), name='b1_1')
z1_1 = tf.matmul(X, W1_1)+b1_1
z1_1 = tf.nn.sigmoid(z1_1)

W2 = tf.Variable(tf.random_normal([2, n]), name='W2')
b2 = tf.Variable(tf.random_normal([n]), name='b2')
Z2 = tf.matmul(z1_1, W2) + b2

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=Z2, labels=Y_onehot)
cost = tf.reduce_mean(cost_i)

hypothesis = tf.nn.softmax(Z2) # pre-defined function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost) # Adam, Adadelta, Momentum,

# prediction = tf.math.argmax(hypothesis, 1)
is_correct = tf.equal(tf.math.argmax(hypothesis, 1), tf.math.argmax(Y_onehot, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        x_data = np.array([[np.random.uniform(-1,1), np.random.uniform(-1,1)] for i in range(128)])
        y_data = np.array([result(u, v) for u, v in x_data])
#         print(x_data)
#         x_data = dataset_x
#         y_data = dataset_z
        sess.run(optimizer, feed_dict ={X: x_data, Y: y_data})
        if step % 100:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            
        x_test = np.array([[np.random.uniform(-1,1), np.random.uniform(-1,1)] for i in range(128)])
        y_test = np.array([result(u, v) for u, v in x_test])
        
        #print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
        #print("GroundTruth:", y_test)
        # print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
        # print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
        
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))