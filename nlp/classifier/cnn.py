import tensorflow as tf

from nlp.classifier.dl_framework import DLGeneralLayer, Instance


class TextCNN(object):
    """
    字符级CNN文本分类
    词嵌入层->卷积层->池化层->softmax层
    """
    # 卷积层和池化层
    # 为3,4,5分别创建128个过滤器，总共3×128个过滤器
    # 过滤器形状为[3,128,1,128]，表示一次能过滤三个字，最后形成188×128的特征向量
    # 池化核形状为[1,188,1,1]，128维中的每一维表示该句子的不同向量表示，
    # 池化即从每一维中提取最大值表示该维的特征
    # 池化得到的特征向量为128维
    def dnn_layer(self, sequence_length, embedding_size,
                  embedded_chars_expanded, filter_sizes, num_filters):
        pooled_outputs = []
        with tf.name_scope('conv'):
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope('conv-%swindow' % filter_size):
                    # 卷积层
                    filter_shape = [filter_size, embedding_size,
                                    1, num_filters]
                    # truncated_normal作用是以正态分布来截取值
                    w = tf.Variable(
                        tf.truncated_normal(filter_shape, stddev=0.1),
                        name='w')
                    conv = tf.nn.conv2d(
                        embedded_chars_expanded,
                        w,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='conv')
                    b = tf.Variable(
                        tf.constant(0.1, shape=[num_filters]), name='b')
                    # ReLU被选为激活函数
                    tf.nn.relu(tf.nn.bias_add(conv, b), name='activation')

                self.distribution_variable.append(('conv/conv-%swindow'
                                                   % filter_size)
                                                  + '/activation:0')
                self.distribution_variable.append(('conv/conv-%swindow'
                                                   % filter_size) + '/w:0')
                self.distribution_variable.append(('conv/conv-%swindow'
                                                   % filter_size) + '/b:0')

        with tf.name_scope('pool'):
            for i, filter_size in enumerate(filter_sizes):

                # for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope('maxpool-%swindow' % filter_size):
                    # 池化层
                    h = tf.get_default_graph(). \
                        get_tensor_by_name('conv/conv-%swindow'
                                           % filter_size + '/activation:0')
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='pool')
                    pooled_outputs.append(pooled)

            h_pool = tf.concat(pooled_outputs, 3)
            num_filters_total = num_filters * len(filter_sizes)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        return num_filters_total, h_pool_flat

    def __init__(self,
                 sequence_length,
                 num_classes,
                 vocab,
                 parameter
                 # l2_reg_lambda=0.0,
                 # lr=1e-3,
                 ):
        l2_reg_lambda = parameter.l2_reg_lambda
        lr = parameter.lr

        self.distribution_variable = []
        embedding_size = vocab.shape[1]

        # 输入, 输出, dropout的占位符
        # placeholder可以理解为是一种常量，而是由用户在调用run方法时传递的，
        # 可理解为一种特别的形参
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name='dropout_keep_prob')

        # l2正则化损失值（可选）
        # l2_loss = tf.constant(0.0)
        # 词嵌入层
        # W为词汇表，大小为0～词汇总数，索引对应不同的字，
        # 每个字映射为128维的数组，比如[3800,128]
        with tf.name_scope('converter'):
            # 开始匹配，将一个文本序列转换为
            w = tf.Variable(
                vocab, dtype=tf.float32,
                name='W')

            embedded_chars = tf.nn.embedding_lookup(w, self.input_x)

            # 为这个K维的词向量在最后一个维度后面再增加一个维度
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        filter_sizes = list(map(int, '3,4,5'.split(',')))
        num_filters = 128
        self.num_filters_total, self.h_pool_flat \
            = self.dnn_layer(sequence_length, embedding_size,
                             embedded_chars_expanded, filter_sizes,
                             num_filters)

        self.connected_layer = DLGeneralLayer(self.h_pool_flat,
                                              self.num_filters_total,
                                              num_classes,
                                              l2_reg_lambda, lr, self.input_y,
                                              self.dropout_keep_prob)
        self.connected_layer.classifier_end_layers()


if __name__ == '__main__':
    single_cnn = Instance(TextCNN)
    single_cnn.run_pipeline()
