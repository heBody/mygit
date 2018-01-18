import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

x = np.linspace(-1, 1, 500, dtype=np.float32)
# y = np.power(x, 3)
y = 1.5*(x **2) + 1.2
# y = 1.5*x + 1.2

noise = np.random.rand(len(x))
y = y + noise

input = np.stack((x, y), axis=-1)

tf_input = tf.placeholder(tf.float32,  [None, 2], "input")

tfx = tf_input[:, :1]
tyf = tf_input[:, 1:]

# l1 = tf.layers.dense(tfx, 20, tf.nn.relu6, name="l1")
l1 = tf.layers.dense(tfx, 10, tf.nn.relu, name="l1")
l2 = tf.layers.dense(l1, 3, tf.nn.relu, name="l2")
# l3 = tf.layers.dense(l2, 10, tf.nn.relu, name="l3")
out = tf.layers.dense(l2, 1)

loss = tf.losses.mean_squared_error(tyf, out)
# loss = tf.reduce_sum(tf.square(out - tyf))
opt = tf.train.GradientDescentOptimizer(0.1)
train_step = opt.minimize(loss)

# model = Weight * (tx**2) + biases
# model = Weight * (tx**2) + W1 * tx + biases
# losses = tf.reduce_sum(tf.square(model - ty))
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(losses)

flg = plt.figure()
ax = flg.add_subplot(1, 1, 1)
# plt.ion()
plt.scatter(x, y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pL = None
    for i in range(100):
        batch_index = np.random.randint(len(input), size=50)
        sess.run(train_step, {tf_input:input[batch_index]})
        prediction = sess.run(out, feed_dict={tf_input:input})  # 计算预测值
        if i%5 ==0:
            # cl, cl1 = sess.run([loss, out], {tf_input:input})
            # print( cl, cl1)
            plt.clf()  # 清空当前图像
            plt.scatter(x, y)
            plt.plot(x, prediction, 'c-', lw='5')
            testVal = random.random()
            testArr = np.array([[testVal, 0]])
            print(testVal, 1.5*(testVal **2) + 1.2, sess.run([out], {tf_input:testArr}))
            # print(testVal, np.power(testArr, 3), sess.run([out], {tf_input:testArr}))
            cl = sess.run([loss], {tf_input: input})
            print(cl)
            plt.pause(0.1)  # 暂停0.1s

plt.show()



# plt.plot(x, cx*(x**2) + cb, "r")
# plt.show()



