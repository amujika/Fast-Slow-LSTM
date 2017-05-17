import tensorflow as tf
import aux

class LN_LSTMCell(tf.contrib.rnn.RNNCell):
    """
    Layer-Norm, with Ortho Initialization and Zoneout.
    https://arxiv.org/abs/1607.06450 - Layer Norm
    https://arxiv.org/abs/1606.01305 - Zoneout
    derived from
    https://github.com/OlavHN/bnlstm
    https://github.com/LeavesBreathe/tensorflow_with_latest_papers
    https://github.com/hardmaru/supercell
    """

    def __init__(self, num_units, f_bias=1.0, use_zoneout=False,
                 zoneout_keep_h = 0.9, zoneout_keep_c = 0.5, is_training = False):
        """Initialize the Layer Norm LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (default 1.0).
          use_recurrent_dropout: float, Whether to use Recurrent Dropout (default False)
          dropout_keep_prob: float, dropout keep probability (default 0.90)
        """
        self.num_units = num_units
        self.f_bias = f_bias

        self.use_zoneout  = use_zoneout
        self.zoneout_keep_h = zoneout_keep_h
        self.zoneout_keep_c = zoneout_keep_c

        self.is_training = is_training

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            h, c = state

            h_size = self.num_units
            x_size = x.get_shape().as_list()[1]

            w_init = aux.orthogonal_initializer(1.0)
            h_init = aux.orthogonal_initializer(1.0)
            b_init = tf.constant_initializer(0.0)

            W_xh = tf.get_variable('W_xh',
                                   [x_size, 4 * h_size], initializer=w_init, dtype=tf.float32)
            W_hh = tf.get_variable('W_hh',
                                   [h_size, 4 * h_size], initializer=h_init, dtype=tf.float32)
            bias = tf.get_variable('bias', [4 * h_size], initializer=b_init, dtype=tf.float32)

            concat = tf.concat(axis=1, values=[x, h])  # concat for speed.
            W_full = tf.concat(axis=0, values=[W_xh, W_hh])
            concat = tf.matmul(concat, W_full) + bias
            concat = aux.layer_norm_all(concat, 4, h_size, 'ln')

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(axis=1, num_or_size_splits=4, value=concat)

            new_c = c * tf.sigmoid(f + self.f_bias) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(aux.layer_norm(new_c, 'ln_c')) * tf.sigmoid(o)

            if self.use_zoneout:
                new_h, new_c = aux.zoneout(new_h, new_c, h, c, self.zoneout_keep_h,
                                           self.zoneout_keep_c, self.is_training)

        return new_h, (new_h, new_c)

    def zero_state(self, batch_size, dtype):
        h = tf.zeros([batch_size, self.num_units], dtype=dtype)
        c = tf.zeros([batch_size, self.num_units], dtype=dtype)
        return (h, c)

