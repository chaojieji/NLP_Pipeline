import os
import datetime
import warnings
import collections
import logging

import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from nlp.ai_toolkit.preprocess import embed
from nlp.ai_toolkit.evaluation import f1_score
from nlp.ai_toolkit.hyperparameter_optimizer import cyclic_learning_rate\
    as clr
from nlp.classifier.question_classifier import QuestionClassfier

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class DLGeneralLayer(object):
    def __init__(self, hs, input_dimension, output_dimension,
                 l2_reg_lambda, lr, y, dropout_keep_prob):
        self.hs = hs
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.l2_reg_lambda = l2_reg_lambda
        self.lr = lr
        self.y = y
        self.dropout_keep_prob = dropout_keep_prob
        self.h_drop = None
        self.correct_predictions = None
        self.predictions = None
        self.loss = None
        self.y_true = None
        self.accuracy = None
        self.global_step = None
        self.optimizer = None
        self.train_op = None
        self.grads_and_vars = None

    def classifier_end_layers(self):
        with tf.name_scope('fc'):
            # dropout
            self.h_drop = tf.nn.dropout(self.hs,
                                        self.dropout_keep_prob)
            # 全连接层
            w = tf.Variable(
                tf.truncated_normal(
                    [self.input_dimension, self.output_dimension], stddev=0.1),
                name='W')
            b = tf.Variable(tf.constant(0.1, shape=[self.output_dimension]),
                            name='b')
            if self.l2_reg_lambda:
                w_l2_loss\
                    = tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)(w)
                # 把变量放入一个集合，把很多变量变成一个列表
                tf.add_to_collection('losses', w_l2_loss)

            # 最后个隐藏状态输出，可用于Embedding Projector
            scores = \
                tf.sigmoid(tf.nn.xw_plus_b(self.h_drop, w, b,
                                           name="linear_regression"),
                           name='scores')
            self.predictions = tf.argmax(scores, 1, name='predictions')

        # 计算交叉损失熵（分支一）
        with tf.name_scope('loss'):
            # reduce_mean: 计算各个维度的平均值
            mse_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=scores, labels=self.y))
            tf.add_to_collection('losses', mse_loss)
            # 将所有维度的值全部加起来，e.g.对[3, 4]做add_n之后的结果为7
            self.loss = tf.add_n(tf.get_collection('losses'))
            self.y_true = tf.argmax(self.y, 1, name="y_true")

        # 正确率（分支二）
        with tf.name_scope('evaluation'):
            # equal：对比这两个矩阵或者向量的相等元素，相等为True，不等为False
            self.correct_predictions = tf.equal(self.predictions,
                                                tf.argmax(self.y, 1))
            # cast：将x的数据格式转化成dtype
            self.accuracy = tf.reduce_mean(
                tf.cast(self.correct_predictions, 'float'), name='accuracy')

        with tf.name_scope('train'):
            # 定义训练相关操作
            # global_step就是训练的次数，可以理解为进行梯度下降的次数
            self.global_step = tf.Variable(0, name='global_step',
                                           trainable=False)
            # 普通的梯度下降优化器
            # self.optimizer = tf.train.AdamOptimizer(lr)
            if type(self.lr) is float:
                self.optimizer = tf.train.AdamOptimizer(self.lr)
            else:
                self.optimizer\
                    = tf.train.GradientDescentOptimizer(
                        clr.cyclic_learning_rate(global_step=self.global_step,
                                                 learning_rate=self.lr[
                                                     "learning_rate"],
                                                 max_lr=self.lr["max_lr"],
                                                 step_size=self.lr[
                                                     "step_size"],
                                                 mode=self.lr["mode"]))
                # clr_lr = clr.cyclic_learning_rate(global_step
                #                                   =self.global_step,
                #                                   learning_rate=self.lr[0],
                #                                   max_lr=self.lr[1],
                #                                   step_size=self.lr[2],
                #                                   mode=self.lr[3])
                # self.optimizer\
                #     = tf.train.AdamOptimizer(learning_rate=clr_lr)

            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

            self.train_op = self.optimizer.apply_gradients(
                self.grads_and_vars, global_step=self.global_step)


class DeepLearningArchitecture(QuestionClassfier):
    def train_step(self, x_batch, y_batch,
                   x_train_origin, generated_text_ph,
                   step_num, dl, tokenizer, parameters, sess,
                   train_summary_op, train_summary_writer,
                   dic_lr_acc):

        """
        一个训练步骤
        """
        feed_dict = {
            dl.input_x: x_batch,
            dl.input_y: y_batch,
            generated_text_ph: tokenizer.sequences_to_texts(x_batch),
            dl.dropout_keep_prob: parameters.dropout_keep_prob
        }

        if self.is_runtime_step(step_num, x_train_origin, parameters):
            # runtime统计
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        # feed_dict参数是给使用placeholder创建出来的tensor赋值
        if type(parameters.lr) is float:
            _, step, summaries, loss, prediction, y_true = sess.run([
                dl.connected_layer.train_op, dl.connected_layer.global_step,
                train_summary_op, dl.connected_layer.loss,
                dl.connected_layer.predictions, dl.connected_layer.y_true
            ], feed_dict, options=run_options, run_metadata=run_metadata)
            time_str = datetime.datetime.now().isoformat()

            # 获取评价指标值
            accuracy, precision, recall, f1, confidence_interval \
                = f1_score.sklearn_get_metrics(y_true, prediction)
            if parameters.verbose:
                log.info('{}: step {}, loss {:g}, acc {:g}, '
                                 'precision {:g}, recall {:g}, f1 {:g}'
                                 .format(time_str, step, loss, accuracy,
                                 precision, recall, f1))
        else:

            _, step, summaries, loss,\
              prediction, y_true, lr_current = sess.run([
                dl.connected_layer.train_op, dl.connected_layer.global_step,
                train_summary_op, dl.connected_layer.loss,
                dl.connected_layer.predictions, dl.connected_layer.y_true,
                dl.connected_layer.optimizer._learning_rate
              ], feed_dict, options=run_options, run_metadata=run_metadata)
            time_str = datetime.datetime.now().isoformat()

            # 获取评价指标值
            accuracy, precision, recall, f1, confidence_interval \
                = f1_score.sklearn_get_metrics(y_true, prediction)
            if parameters.verbose:
                log.info('{}: step {}, loss {:g}, acc {:g}, precision'
                                 ' {:g}, recall {:g}, f1 {:g}, lr {:g}'
                                 .format(time_str, step, loss, accuracy,
                                 precision, recall, f1, lr_current))
            dic_lr_acc[lr_current] = accuracy

        if len(summaries) > 0:
            train_summary_writer.add_summary(summaries, step)

        if self.is_runtime_step(step_num, x_train_origin, parameters):
            train_summary_writer.add_run_metadata(run_metadata,
                                                  'step%d' % step_num)

    @staticmethod
    def projector_embedding(x_train, y_train, x_train_origin,
                            y_train_origin, generated_text_ph, dl, tokenizer,
                            sess, embedding_hs, dir_projector, list_label,
                            step):

        generated_texts = tokenizer.sequences_to_texts(x_train)
        feed_dict = {
            dl.input_x: x_train,
            dl.input_y: y_train,
            generated_text_ph: generated_texts,
            dl.dropout_keep_prob: 1.0
        }
        projector_saver = tf.train.Saver([tf.get_default_graph()
                                         .get_tensor_by_name(
            'Summary/embedding_hs:0')])
        sess.run(embedding_hs, feed_dict)
        # 注意一定要按如下格式输出（step必须存在），否则会出现tensorboard无法定位数据！
        projector_saver.save(sess, os.path.join(dir_projector), step)

        dir_tmp = os.path.join(dir_projector, 'metadata.tsv')
        with open(dir_tmp, 'w', encoding="utf-8") as f:
            f.write("Index\tLabel\n")
            for i in range(len(x_train)):
                f.write(("%s\t%s\n" % (x_train_origin[i],
                                       list_label[int(y_train_origin[i])])))
                pass

    @staticmethod
    def dev_step(x_batch, y_batch, dl, sess, dev_summary_op, writer):
        """
        在开发集上验证模型
        """
        feed_dict = {
            dl.input_x: x_batch,
            dl.input_y: y_batch,
            dl.dropout_keep_prob: 1.0
        }
        step, summaries, prediction, y_true = sess.run(
            [dl.connected_layer.global_step, dev_summary_op,
             dl.connected_layer.predictions, dl.connected_layer.y_true],
            feed_dict)
        # 获取评价指标值
        accuracy, precision, recall, f1, confidence_interval \
            = f1_score.sklearn_get_metrics(y_true, prediction)
        log.info("accuracy = %.3f, precision = %.3f, recall = %.3f,"
                         " f1 = %.3f, confidence_interval = %s"
                         % (accuracy, precision, recall, f1,
                            str(confidence_interval)))
        writer.add_summary(summaries, step)
        return f1

    def text_summary(self, generated_text_ph, correct_predictions, size):
        sample_pass = self.pick_sample(generated_text_ph, correct_predictions,
                                       True, size)

        sample_fail = self.pick_sample(generated_text_ph, correct_predictions,
                                       False, size)
        text_pass = tf.summary.text('Pass:', sample_pass)
        text_fail = tf.summary.text('Fail:', sample_fail)
        text_merged = tf.summary.merge([text_pass, text_fail])
        return text_merged

    @staticmethod
    def projector_summary(model, sample_num, parameters):
        embedding_size = model.connected_layer.input_dimension
        embedding_input = model.h_pool_flat
        log_dir = parameters.model_dump_path + "summaries/" + "train"

        # 只是创建了一个文件
        writer = tf.summary.FileWriter(log_dir)
        # 667
        embedding = tf.Variable(tf.zeros([sample_num, embedding_size]),
                                name="embedding_hs")
        embedding_hs = embedding.assign(embedding_input[:sample_num])

        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = embedding.name
        # embedding_config.tensor_name = embedding.name

        embedding_config.metadata_path =\
            os.path.abspath(os.path.join(log_dir, 'metadata.tsv'))
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer,
                                                                      config)
        return embedding_hs, log_dir

    @staticmethod
    def scalar_summary(loss, accuracy, lr, gradient):
        scalar = []
        # 损失值和正确率的摘要
        if loss is not None:
            scalar.append(tf.summary.scalar('loss', loss))
        if accuracy is not None:
            scalar.append(tf.summary.scalar('accuracy', accuracy))
        if lr is not None:
            scalar.append(tf.summary.scalar('lr', lr))

        # 变量的梯度值以及这些值的离散性
        if gradient is not None:
            for g, v in gradient:
                if g is not None:
                    # 对tf.summary.scalr和histpgram的作用解释：
                    # 生成【变量】的监控信息，并将生成的监控信息写入【日志文件】，
                    # 注意不只是生成，还包括写入！
                    grad_value_summary = tf.summary.histogram(
                        '{}/grad/value'.format(v.name), g)
                    scalar.append(grad_value_summary)
                    # zero_fraction函数的作用是返回：0在value中的小数比例
                    # 越高证明0的单元越多
                    sparsity_summary\
                        = tf.summary.scalar('{}/grad/sparsity'
                                            .format(v.name),
                                            tf.nn.zero_fraction(g))
                    scalar.append(sparsity_summary)

        scalar_merged = tf.summary.merge(scalar)
        return scalar_merged

    # 决定tensorboard中distribution和histogram面板效果
    @staticmethod
    def distribution_summary(distribution_variable):
        distribution = []

        for item in distribution_variable:
            variable = tf.get_default_graph().get_tensor_by_name(item)
            distribution.append(tf.summary.histogram(
                item, variable))

        distribution_merged = []
        if len(distribution) > 0:
            distribution_merged = tf.summary.merge(distribution)
        return distribution_merged

    # 保证每个批次的数据都不同
    @staticmethod
    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        """
        生成一个batch迭代器
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_idx:end_idx]

    @staticmethod
    def tokenize_text(x_train_origin, x_valid_origin, x_test_origin,
                      tokenizer, max_sequence_length):
        list_sample = []
        for item in [x_train_origin, x_valid_origin, x_test_origin]:
            sequences = tokenizer.texts_to_sequences(item)

            if max_sequence_length > 0:
                # 补齐长度,最终用于输入模型的文本
                item = pad_sequences(sequences, maxlen=max_sequence_length)
            else:
                item = np.array(sequences)

            list_sample.append(item)
        x_train, x_valid, x_test = list_sample[0],\
            list_sample[1], list_sample[2]
        return x_train, x_valid, x_test

    @staticmethod
    def pick_sample(sample, actual_result, target_result, batch_size):
        tf.constant(target_result, shape=[batch_size])
        mapped_sample = tf.equal(actual_result, target_result)
        mapped_sample = tf.boolean_mask(sample, mapped_sample)
        return mapped_sample

    @staticmethod
    def generate_summary_writer(sess, out_dir, parameter):
        summary_dir = os.path.join(out_dir, 'summaries', parameter)
        summary_writer = tf.summary.FileWriter(summary_dir,
                                               sess.graph)
        return summary_writer

    @staticmethod
    def del_all_flags(parameters):
        for keys in [keys for keys in parameters._flags()]:
            parameters.__delattr__(keys)

    def parameter_definition(self):
        self.del_all_flags(tf.flags.FLAGS)
        flags = tf.flags
        flags.DEFINE_boolean('verbose', True,
                             'print performance during training')

        # 嵌入式向量维度
        flags.DEFINE_integer('embedding_dim', 300,
                             'Dimensionality of embedding (default: 128)')

        flags.DEFINE_string('model_dump_path', '../../data/model/',
                            'Path to save model')

        # Dropout率
        flags.DEFINE_float('dropout_keep_prob', 0.5,
                           'Dropout keep probability (default: 0.5)')
        # l2正则系数
        flags.DEFINE_float('l2_reg_lambda', 0.0,
                           'L2 regularization lambda (default: 0.0)')

        # 学习率
        flags.DEFINE_float('lr', 1e-3, 'Learning Rate (default: 1e-3)')
        # 批次大小
        flags.DEFINE_integer('batch_size', 64, 'Batch Size (default: 64)')
        # epoch次数
        flags.DEFINE_integer('num_epochs', 20000,
                             'Number of training epochs (default: 10)')
        # 多少个样本进行一次评价
        flags.DEFINE_integer(
            'evaluate_every', 100,
            'Evaluate model on dev set after this many steps (default: 100)')

        # 每100步保存一次
        flags.DEFINE_integer('checkpoint_every', 100,
                             'Save model after this many steps (default: 100)')
        # 不分好坏就是保存最后的num_checkpoints个模型
        flags.DEFINE_integer('num_checkpoints', 1,
                             'Number of checkpoints to store (default: 1)')
        # 其他参数
        # =====
        # allow_soft_placement: 获取到 operations 和 Tensor
        # 被指派到哪个设备(几号CPU或几号GPU)上运行,会在终端打印出各项操作是在哪个设备上运行
        flags.DEFINE_boolean('allow_soft_placement', True,
                             'Allow device soft device placement')
        # log_device_placement: 允许tf自动选择一个存在并且可用的设备来运行操作，
        # 在多个CPU或GPU设备的时候很有用
        flags.DEFINE_boolean('log_device_placement', False,
                             'Log placement of ops on devices')

        flags.DEFINE_integer('hidden_dimension', 512,
                             'number of value in final neural layer')

        parameters = flags.FLAGS
        dict_parameter = parameters.flag_values_dict()
        log.info('\nParameters:')
        for item in dict_parameter:
            log.info('{}={}'.format(item, dict_parameter[item]))
        return parameters

    @staticmethod
    def init_file(parameters):
        if tf.gfile.Exists(parameters.model_dump_path + "/summaries"):
            tf.gfile.DeleteRecursively(parameters.model_dump_path
                                       + "/summaries")
        tf.gfile.MakeDirs(parameters.model_dump_path + "/summaries")
        if tf.gfile.Exists(parameters.model_dump_path + "/checkpoints"):
            tf.gfile.DeleteRecursively(parameters.model_dump_path
                                       + "/checkpoints")
        tf.gfile.MakeDirs(parameters.model_dump_path + "/checkpoints")
        pass

    def init_saver_file(self, sess, parameters):
        # 目录定义阶段
        # =====
        # 父目录位置
        # out_dir = os.path.abspath(
        #     os.path.join(os.path.curdir, 'runs'))
        out_dir = parameters.model_dump_path
        # 训练和确认过程生成的记录
        train_summary_writer = self.generate_summary_writer(sess, out_dir,
                                                            "train")
        dev_summary_writer = self.generate_summary_writer(sess, out_dir, "dev")

        # 整个模型的保存目录
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        return train_summary_writer, dev_summary_writer, checkpoint_prefix

    def generate_summary(self, model, generated_text_ph, parameters,
                         sample_num, distribution_variable):
        with tf.name_scope('Summary'):
            list_text_summary\
                = self.text_summary(generated_text_ph,
                                    model.connected_layer.correct_predictions,
                                    parameters.batch_size)
            list_distribution_summary\
                = self.distribution_summary(distribution_variable)
            list_basic_scalar_summary\
                = self.scalar_summary(model.connected_layer.loss,
                                      model.connected_layer.accuracy,
                                      None, None)
            if type(parameters.lr) is dict:
                lr = model.connected_layer.optimizer._learning_rate
            else:
                lr = model.connected_layer.optimizer._lr
            list_advance_scalar_summary\
                = self.scalar_summary(None, None, lr,
                                      model.connected_layer.grads_and_vars)
            if list_distribution_summary is not []:
                train_summary_op = list_text_summary\
                                   + list_basic_scalar_summary\
                                   + list_advance_scalar_summary
            else:
                train_summary_op = list_text_summary\
                                   + list_basic_scalar_summary\
                                   + list_advance_scalar_summary\
                                   + list_distribution_summary
            dev_summary_op = list_basic_scalar_summary
            embedding_hs, dir_projector\
                = self.projector_summary(model, sample_num, parameters)
            return train_summary_op, dev_summary_op,\
                embedding_hs, dir_projector

    @staticmethod
    def is_runtime_step(step_num, x_train_origin, parameters):
        return step_num % int(2 * len(x_train_origin) / parameters.batch_size)\
               == 0 and step_num > 1

    def data_transformer(self, embed_path, max_sequence_length, max_vocab):
        self.embed, self.x_train_origin, self.y_train,\
          self.x_valid_origin, self.y_valid, self.x_test_origin, \
          self.y_test, EMBEDDING_DIM, list_label, _, _, _,\
          self.tokenizer, self.y_train_origin, self.y_valid_origin,\
          self.y_test_origin\
          = embed.sequence_tokenize(self.lst_label, self.x_train,
                                    self.y_train, self.x_valid,
                                    self.y_valid, self.x_test,
                                    self.y_test, embed_path, max_vocab)
        self.x_train, self.x_valid, self.x_test\
            = self.tokenize_text(self.x_train_origin, self.x_valid_origin,
                                 self.x_test_origin, self.tokenizer,
                                 max_sequence_length)

    def define_and_run(self, parameters, x, x_text, y, y_text,
                       x_valid, y_valid):
        self.init_file(parameters)
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=parameters.allow_soft_placement,
                log_device_placement=parameters.log_device_placement)

            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # generated_text_ph: 实际文本内容
                generated_text_ph = tf.placeholder(tf.string, shape=(None,),
                                                   name='generated_text')
                dl = self.implemented_dl(
                    sequence_length=x.shape[1], num_classes=y.shape[1],
                    vocab=self.embed,
                    parameter=parameters
                )
                train_summary_op, dev_summary_op, embedding_hs, dir_projector \
                    = self.generate_summary(dl, generated_text_ph, parameters,
                                            len(self.x_train_origin),
                                            dl.distribution_variable)

                train_summary_writer, dev_summary_writer, checkpoint_prefix \
                    = self.init_saver_file(sess, parameters)

                # 生成batches
                batches = self.batch_iter(
                    list(zip(x, y, x_text, y_text)),
                    parameters.batch_size, parameters.num_epochs)
                saver = tf.train.Saver(
                    tf.global_variables(),
                    max_to_keep=parameters.num_checkpoints)

                # 初始化变量
                sess.run(tf.global_variables_initializer())

                dic_lr_acc = collections.OrderedDict()
                best_score = 0
                # 迭代训练每个batch
                for batch in batches:
                    x_batch, y_batch, x_batch_origin,\
                      y_batch_origin = zip(*batch)
                    current_step\
                        = tf.train.global_step(
                          sess, dl.connected_layer.global_step)
                    self.train_step(x_batch, y_batch,
                                    x_text, generated_text_ph,
                                    current_step, dl, self.tokenizer,
                                    parameters, sess, train_summary_op,
                                    train_summary_writer, dic_lr_acc)
                    if current_step % int(len(x_text) / parameters.batch_size)\
                            == 0 and current_step > 1:
                        log.info('\nEvaluation:')
                        cur_score = self.dev_step(x_valid, y_valid, dl, sess,
                                                  dev_summary_op,
                                                  dev_summary_writer)
                        log.info("Best score in history is "
                                         + str(best_score))

                        if cur_score > best_score:
                            path = saver.save(
                                sess, checkpoint_prefix)
                            log.info('Saved model checkpoint to {}\n'
                                             .format(path))
                            best_score = cur_score
                            self.projector_embedding(x, y,
                                                     x_text, y_text,
                                                     generated_text_ph,
                                                     dl, self.tokenizer, sess,
                                                     embedding_hs,
                                                     dir_projector,
                                                     self.lst_label,
                                                     current_step)
                if type(parameters.lr) is not float:
                    self.run_clr(dic_lr_acc)

    def fit(self, **kwargs):
        parameters = kwargs["parameter"]
        parameters.num_epochs = kwargs["max_iter"]
        self.define_and_run(parameters, kwargs["x"], kwargs["x_text"],
                            kwargs["y"], kwargs["y_text"], kwargs["x_valid"],
                            kwargs["y_valid"])
        self.load_tf_model([parameters.model_dump_path + "checkpoints/",
                            "model.meta"])
        return self.evaluation(kwargs["x_valid"], kwargs["y_valid_text"])

    def load_tf_model(self, saver_path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver\
                = tf.train.import_meta_graph(saver_path[0] + saver_path[1],
                                             clear_devices=True)
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess,
                                   tf.train.latest_checkpoint(saver_path[0]))

    def __init__(self, implemented_dl):
        warnings.filterwarnings('ignore')
        # zh.set_chinese_font()
        QuestionClassfier.__init__(self)
        self.clf = self
        self.implemented_dl = implemented_dl
        self.x_train_origin = None
        self.x_valid_origin = None
        self.x_test_origin = None
        self.tokenizer = None
        self.y_train_origin = None
        self.y_valid_origin = None
        self.y_test_origin = None
        self.sess = None
        self.predictions = None
        self.scores = None
        self.saver = None
        self.graph = None

    def predict(self, x):
        self.predictions = self.graph.get_tensor_by_name("fc/predictions:0")
        self.scores = self.graph.get_tensor_by_name("fc/scores:0")
        feed_dict = {
            "input_x:0": x,
            "dropout_keep_prob:0": 1.0
        }

        prediction, prediction_pro = self.sess.run(
            [self.predictions, self.scores],
            feed_dict)

        # 非深度学习模型不需要进行下面的转换
        lst_predicted_label = []
        for item in prediction:
            lst_predicted_label.append(self.lst_label[item])

        if len(lst_predicted_label) == 1:
            lst_predicted_label = lst_predicted_label[0]
            prediction_pro = max(prediction_pro[0])

        return lst_predicted_label, prediction_pro

    def predict_proba(self, x_input):
        self.scores = self.graph.get_tensor_by_name("fc/scores:0")
        feed_dict = {
            "input_x:0": x_input,
            "dropout_keep_prob:0": 1.0
        }

        scores = self.sess.run(
            [self.scores],
            feed_dict)
        return scores[0]


class Instance(object):
    def __init__(self, dl):
        self.clf = dl

    def model_selection(self):
        predictor = DeepLearningArchitecture(self.clf)
        # 参数定义
        parameters = predictor.parameter_definition()
        parameters.batch_size = 32
        # parameters.lr = 0.0001
        # parameters.lr = 0.01
        # parameters.dropout_keep_prob = 1.0

        # parameters = copy.deepcopy(dict(parameters))
        return predictor, parameters

    @staticmethod
    def data_cv_split(predictor, tv_rate, v_t_rate):
        from nlp.classifer_visualization import visualizer

        predictor.embed_mode = "w2v-avg"
        predictor.data = visualizer.get_data("../../data/query_origin_char",
                                            ["意图", "问题"], '@', '$')
        # predictor.des_visualization(["意图", "问题"])
        # predictor.dr_visualization(2, "意图", 100, "问题")

        predictor.split_data("问题", "意图", 1.0 - float(tv_rate), v_t_rate,
                             "../../data/Xiaodou_business_origin_char_split")

        # [predictor.lst_label, [predictor.x_train, predictor.y_train],
        #  [predictor.x_valid, predictor.y_valid],
        #  [predictor.x_test, predictor.y_test]] \
        #     = predictor \
        #     .load_model("../../data/Xiaodou_business_origin_char_split.pkl")
        total_sample_num = len(predictor.x_train)
        return total_sample_num

    @staticmethod
    def data_transformer(predictor, max_sequence, max_vob,
                         path="../../data/sgns.target.word-character.char1-2"
                              ".dynwin5.thr10.neg5.dim300.iter5"):
        predictor.data_transformer(path,
                                   max_sequence, max_vob)

    @staticmethod
    def train_model(predictor, parameters, iter_num):
        report = predictor.train(max_iter=iter_num, x=predictor.x_train,
                                 x_text=predictor.x_train_origin,
                                 y=predictor.y_train,
                                 y_text=predictor.y_train_origin,
                                 x_valid=predictor.x_valid,
                                 y_valid=predictor.y_valid,
                                 y_valid_text=predictor.y_valid_origin,
                                 saver_path=["../../data/model/checkpoints/",
                                             "model.meta"],
                                 parameter=parameters)
        return report

    @staticmethod
    def evaluate_model(predictor):
        predictor.load_tf_model(["../../data/model/checkpoints/",
                                 "model.meta"])
        predictor.embed_mode = "w2v"
        predictor.evaluation(predictor.x_train, predictor.y_train_origin)
        predictor.evaluation(predictor.x_valid, predictor.y_valid_origin)
        predictor.evaluation(predictor.x_test, predictor.y_test_origin)

    def run_pipeline(self):
        predictor, parameters = self.model_selection()
        self.data_cv_split(predictor, 0.7, 0.5)
        self.data_transformer(predictor, 20, 1000)
        self.train_model(predictor, parameters, 10)
        self.evaluate_model(predictor)
