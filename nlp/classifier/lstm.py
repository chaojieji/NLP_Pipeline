import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from nlp.classifier.dl_framework import DLGeneralLayer, Instance

class lstm(object):
    def dnn_layer(self, x):

        with tf.name_scope('rnn'):
            rnn_cell = rnn.BasicLSTMCell(self.num_filters_total,
                                         reuse=tf.AUTO_REUSE)
            initial_state = rnn_cell.zero_state(tf.shape(self.input_x)[0],
                                                tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(rnn_cell, x,
                                           initial_state=initial_state,
                                           dtype=tf.float32)
            self.h_pool_flat = tf.transpose(outputs, [1, 0, 2])[-1]

    def __init__(self,
                 sequence_length,
                 num_classes,
                 vocab,
                 parameter
                 ):
        l2_reg_lambda = parameter.l2_reg_lambda
        lr = parameter.lr
        self.num_filters_total = parameter.hidden_dimension
        self.distribution_variable = []

        # 输入, 输出, dropout的占位符
        # placeholder可以理解为是一种常量，而是由用户在调用run方法时传递的，
        # 可理解为一种特别的形参
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name='dropout_keep_prob')

        with tf.name_scope('converter'):
            # 开始匹配，将一个文本序列转换为
            w = tf.Variable(
                vocab, dtype=tf.float32,
                name='W')
            embedded_chars = tf.nn.embedding_lookup(w, self.input_x)

        self.dnn_layer(embedded_chars)
        self.connected_layer = DLGeneralLayer(self.h_pool_flat,
                                              self.num_filters_total,
                                              num_classes,
                                              l2_reg_lambda, lr, self.input_y,
                                              self.dropout_keep_prob)
        self.connected_layer.classifier_end_layers()


if __name__ == '__main__':
    single_lstm = Instance(lstm)
    single_lstm.run_pipeline()
