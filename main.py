# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import time
import sys

import tensorflow as tf

import aux
import reader
import configs
import FSLSTM

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "enwik8",
    "A type of model. Check configs file to know which models are available.")
flags.DEFINE_string("data_path", 'data/',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", 'models/',
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        emb_size = config.embed_size
        F_size = config.cell_size
        S_size = config.hyper_size
        vocab_size = config.vocab_size

        self._initial_state = tf.zeros([batch_size,
                                        F_size * 2
                                        + S_size * 2], dtype=data_type())

        # emb_init = tf.random_uniform_initializer(minval=-config.init_scale,
        #                                         maxval= config.init_scale, dtype=data_type())
        emb_init = aux.orthogonal_initializer(1.0)
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, emb_size], initializer=emb_init, dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.in_k_prob < 1:
            inputs = tf.nn.dropout(inputs, config.in_k_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.

        outputs = []

        F_state = tf.slice(self._initial_state, [0, 0],
                           [-1, config.cell_size * 2])
        S_state = tf.slice(self._initial_state, [0, config.cell_size * 2],
                           [-1, config.hyper_size * 2])

        F_state = tf.split(axis=1, num_or_size_splits=2, value=F_state)
        S_state = tf.split(axis=1, num_or_size_splits=2, value=S_state)

        F1 = FSLSTM.LN_LSTMCell(F_size, use_zoneout=True, use_out_drop=True, is_training=is_training,
                                zoneout_keep_h=config.zoneout_h, zoneout_keep_c=config.zoneout_c,
                                out_keep_prob=config.out_k_prob)
        S = FSLSTM.LN_LSTMCell( S_size, use_zoneout=True, use_out_drop=True, is_training=is_training,
                                zoneout_keep_h=config.zoneout_h, zoneout_keep_c=config.zoneout_c,
                                out_keep_prob=config.out_k_prob)
        F2 = FSLSTM.LN_LSTMCell(F_size, use_zoneout=True, use_out_drop=True, is_training=is_training,
                                zoneout_keep_h=config.zoneout_h, zoneout_keep_c=config.zoneout_c,
                                out_keep_prob=config.out_k_prob)

        print('generating graph')
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()

                F_out, F_state = F1(inputs[:, time_step, :], F_state, 'F1')
                S_out, S_state = S( F_out, S_state, 'S')
                F_out, F_state = F2(S_out, F_state, 'F2')

                outputs.append(F_out)
        print('graph generated')
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, F_size])
        # out_init = tf.random_uniform_initializer(minval=-config.init_scale,
        #                                         maxval= config.init_scale, dtype=data_type())
        out_init = aux.orthogonal_initializer(1.0)
        softmax_w = tf.get_variable(
            "softmax_w", [F_size, vocab_size], initializer=out_init, dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        F_state = tf.concat(axis=1, values=[F_state[0], F_state[1]])
        S_state = tf.concat(axis=1, values=[S_state[0], S_state[1]])
        self._final_state = tf.concat(axis=1, values=(F_state, S_state))

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(cost, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N),
            config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        feed_dict[model.initial_state] = state

        vals = session.run(fetches, feed_dict)

        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, costs / iters,
                   iters * model.input.batch_size / (time.time() - start_time)))

        sys.stdout.flush()

    return costs / (iters * 0.69314718056)




def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    config       = configs.get_config(FLAGS.model)
    eval_config  = configs.get_config(FLAGS.model)
    valid_config = configs.get_config(FLAGS.model)
    print(config.batch_size)
    eval_config.batch_size = 1
    valid_config.batch_size = 20

    raw_data = reader.ptb_raw_data(FLAGS.data_path + config.dataset + '/')
    train_data, valid_data, test_data, _ = raw_data


    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config, input_=valid_input)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config,
                                 input_=test_input)

        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            previous_val = 9999
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.4f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.4f" % (i + 1, valid_perplexity))
                sys.stdout.flush()

                if valid_perplexity < previous_val:
                    print("Storing weights")
                    saver.save(session, FLAGS.save_path + 'model.ckpt')
                    previous_val = valid_perplexity
                elif config.dataset == 'enwik8':
                    config.learning_rate *= 0.1

            print("Loading best weights")
            saver.restore(session, FLAGS.save_path + 'model.ckpt')
            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.4f" % test_perplexity)
            sys.stdout.flush()
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
