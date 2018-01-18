import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

x = np.linspace(-1, 1, 100, dtype=np.float32)
y = 1.5*(x **2) + 1.2

noise = np.random.rand(len(x))
y = y + noise

Weight = tf.Variable([0.3], dtype=tf.float32)
W1 = tf.Variable([0.1], dtype=tf.float32)
biases = tf.Variable([0.5] , dtype=tf.float32)

tx = tf.placeholder(tf.float32)
ty = tf.placeholder(tf.float32)

# model = Weight * (tx**2) + biases
model = Weight * (tx) + W1 * tx + biases
# model = Weight * (tx ** 2) + W1 * tx + biases
losses = (tf.reduce_sum(tf.nn.sigmoid(tf.square(model - ty))))
# losses = tf.reduce_sum(tf.square(model - ty))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(losses)

flg = plt.figure()
ax = flg.add_subplot(1, 1, 1)
# plt.ion()
plt.scatter(x, y)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    pL = None
    for i in range(1000):
        sess.run(train_step, {tx:x, ty:y})
        if(i%50 ==0):
            cx, w1, cb, cl = sess.run([Weight, W1, biases, losses], {tx: x, ty: y})
            # plt.cla()
            if pL != None:
                ax.lines.remove(pL[0])
                pass
            pL = ax.plot(x, cx * (x ** 2) + x * w1 + cb, "r")
            print(cl)

            # testVal = random.random()
            # testArr = np.array([[testVal, 0]])
            # print(testVal, 1.5 * (testVal ** 2) + 1.2, sess.run([out], {tf_input: testArr}))

            plt.pause(0.2)


print(cx, w1, cb, cl)

plt.show()



# plt.plot(x, cx*(x**2) + cb, "r")
# plt.show()



