import tensorflow as tf

class FSRNNCell(tf.contrib.rnn.RNNCell):
    def __init__(self, fast_cells, slow_cell, keep_prob=1.0, training=True):
        """Initialize the basic Fast-Slow RNN.
            Args:
              fast_cells: A list of RNN cells that will be used for the fast RNN.
                The cells must be callable, implement zero_state() and all have the
                same hidden size, like for example tf.contrib.rnn.BasicLSTMCell.
              slow_cell: A single RNN cell for the slow RNN.
              keep_prob: Keep probability for the non recurrent dropout. Any kind of
                recurrent dropout should be implemented in the RNN cells.
              training: If False, no dropout is applied.
        """

        self.fast_layers = len(fast_cells)
        assert self.fast_layers >= 2, 'At least two fast layers are needed'
        self.fast_cells = fast_cells
        self.slow_cell = slow_cell
        self.keep_prob = keep_prob
        if not training: self.keep_prob = 1.0

    def __call__(self, inputs, state, scope='FS-RNN'):
        F_state = state[0]
        S_state = state[1]

        with tf.variable_scope(scope):
            inputs = tf.nn.dropout(inputs, self.keep_prob)

            with tf.variable_scope('Fast_0'):
                F_output, F_state = self.fast_cells[0](inputs, F_state)
            F_output_drop = tf.nn.dropout(F_output, self.keep_prob)

            with tf.variable_scope('Slow'):
                S_output, S_state = self.slow_cell(F_output_drop, S_state)
            S_output_drop = tf.nn.dropout(S_output, self.keep_prob)

            with tf.variable_scope('Fast_1'):
                F_output, F_state = self.fast_cells[1](S_output_drop, F_state)

            for i in range(2, self.fast_layers):
                with tf.variable_scope('Fast_' + str(i)):
                    # Input cannot be empty for many RNN cells
                    F_output, F_state = self.fast_cells[i](F_output[:, 0:1] * 0.0, F_state)

            F_output_drop = tf.nn.dropout(F_output, self.keep_prob)
            return F_output_drop, (F_state, S_state)


    def zero_state(self, batch_size, dtype):
        F_state = self.fast_cells[0].zero_state(batch_size, dtype)
        S_state = self.slow_cell.zero_state(batch_size, dtype)

        return (F_state, S_state)