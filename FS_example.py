import FSRNN
import tensorflow as tf
import numpy as np

#Create one Slow and three Fast cells
slow = tf.contrib.rnn.BasicLSTMCell(100)
fast = [tf.contrib.rnn.BasicLSTMCell(100),
        tf.contrib.rnn.BasicLSTMCell(100),
        tf.contrib.rnn.BasicLSTMCell(100)]

#Create a single FS-RNN using the cells
fs_lstm = FSRNN.FSRNNCell(fast, slow)

#Get initial state and create tf op to run one timestep
init_state = fs_lstm.zero_state(10, tf.float32)
output, final_state = fs_lstm(np.zeros((10, 10), np.float32), init_state)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(output).shape)



