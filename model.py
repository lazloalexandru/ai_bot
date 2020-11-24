import tensorflow.compat.v1 as tf


class Model:
    def __init__(self, num_states, num_actions, params):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = params['BATCH_SIZE']
        self._MAX_CHECKPOINTS = params['MAX_CHECKPOINTS']

        self.path = None

        # define the placeholders
        self._states = None
        self._actions = None

        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None

        self._saver = None

        # now setup the model
        self._define_model()

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 3000, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 3000, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 3000, activation=tf.nn.relu)
        fc4 = tf.layers.dense(fc3, 3000, activation=tf.nn.relu)
        fc5 = tf.layers.dense(fc4, 3000, activation=tf.nn.relu)
        fc6 = tf.layers.dense(fc5, 3000, activation=tf.nn.relu)
        fc7 = tf.layers.dense(fc6, 3000, activation=tf.nn.relu)
        fc8 = tf.layers.dense(fc7, 3000, activation=tf.nn.relu)
        fc9 = tf.layers.dense(fc8, 3000, activation=tf.nn.relu)
        fc10 = tf.layers.dense(fc9, 3000, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc10, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

        self._saver = tf.train.Saver(max_to_keep=self._MAX_CHECKPOINTS)

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states: state.reshape(1, self.num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

    def save(self, sess, step):
        self._saver.save(sess, "checkpoints\\my_model", global_step=step)

    def restore(self, sess):
        res = False

        self.path = tf.train.latest_checkpoint('checkpoints\\')
        print("Checkpoint: ", self.path)
        if self.path is None:
            sess.run(self.var_init)
        else:
            self._saver.restore(sess, self.path)
            res = True

        return res

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init
