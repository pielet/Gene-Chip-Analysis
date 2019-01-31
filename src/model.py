import tensorflow as tf


def kaiming(shape, dtype, partition_info=None):
    return tf.truncated_normal(shape, dtype=dtype)*tf.sqrt(2/float(shape[0]))


class GeneModel(object):

    def __init__(self, learning_rate, use_res, use_dropout, use_norm):
        self.use_res = use_res
        self.use_dropout = use_dropout
        self.use_norm = use_norm
        self.input_size = 846
        self.output_size = 41
        self.linear_size = 1024
        self.dropout_rate = tf.placeholder(tf.float32)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        decay_steps = 100000
        decay_rate = 0.96
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, decay_steps, decay_rate)

        self.build_model()

    def build_model(self):
        with tf.variable_scope('input'):
            self.input = tf.placeholder(tf.float32, shape=[None, self.input_size], name='input')
            self.gt = tf.placeholder(tf.float32, shape=[None, self.output_size], name='ground_truth')

        with tf.variable_scope('linear_model', initializer=kaiming):
            # 846 -> 1024
            w_input = tf.get_variable('w_input', shape=[self.input_size, self.linear_size])
            b_input = tf.get_variable('b_input', shape=[self.linear_size])
            if self.use_norm:
                w_input = tf.clip_by_norm(w_input, 1)   # L1 norm
            y = tf.matmul(self.input, w_input) + b_input
            y = tf.nn.relu(y)
            if self.use_dropout:
                y = tf.nn.dropout(y, self.dropout_rate)

            # two residual blocks
            y = self.res_block(y, self.use_res)

            # 1024 -> 41
            w_output = tf.get_variable('w_output', shape=[self.linear_size, self.output_size])
            b_output = tf.get_variable('b_output', shape=[self.output_size])
            if self.use_norm:
                w_output = tf.clip_by_norm(w_output, 1)
            y = tf.matmul(y, w_output) + b_output

            self.output = y
            self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.gt, y))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, self.global_step)

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

    def res_block(self, input_layer, use_res):
        with tf.variable_scope('res', initializer=kaiming):
            w1 = tf.get_variable('w1', shape=[self.linear_size, self.linear_size])
            b1 = tf.get_variable('b1', shape=[self.linear_size])
            if self.use_norm:
                w1 = tf.clip_by_norm(w1, 1)
            y = tf.matmul(input_layer, w1) + b1
            y = tf.nn.relu(y)
            if self.use_dropout:
                y = tf.nn.dropout(y, self.dropout_rate)

            w2 = tf.get_variable('w2', shape=[self.linear_size, self.linear_size])
            b2 = tf.get_variable('b2', shape=[self.linear_size])
            if self.use_norm:
                w2 = tf.clip_by_norm(w2, 1)
            y = tf.matmul(y, w2) + b2
            y = tf.nn.relu(y)
            if self.use_dropout:
                y = tf.nn.dropout(y, self.dropout_rate)

        if use_res:
            return y + input_layer
        else:
            return y

    def step(self, sess, x, y, is_train=True):
        if is_train:
            return sess.run([self.global_step, self.output, self.loss, self.optimizer],
                            {self.input: x, self.gt: y, self.dropout_rate: 0.5})

        if not is_train:
            return sess.run([self.output], {self.input: x, self.gt: y, self.dropout_rate: 1.0})
