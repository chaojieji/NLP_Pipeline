from abc import abstractmethod
import logging as log

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from nlp.ai_toolkit.preprocess import cv_producer, embed
from nlp.ai_toolkit.evaluation import f1_score
from nlp import io


class QuestionClassfier(object):
    def __init__(self, data=None):
        if data is not None:
            self.data = data
        self.embed = None
        self.lst_label = None
        self.x_train = None
        self.x_valid = None
        self.x_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
        self.clf = None
        self.embed_mode = "bow"
        self.actual_model = None
        self.data = None

    @classmethod
    def get_classifier(cls, algorithm):
        if algorithm == "logistic":
            return LogisticRegressionClassifier()

    def split_data(self, feature_name, label_name, train_untrain_rate,
                   valid_test_rate, str_path=None):
        self.x_train, self.y_train, self.x_valid, self.y_valid, \
            self.x_test, self.y_test, self.lst_label \
            = cv_producer.dump_dataset(self.data, feature_name,
                                       label_name, train_untrain_rate,
                                       valid_test_rate, str_path)

    def save_model(self, fname):
        try:
            model_fname = fname + "_model.txt"
            io.save(self.clf, model_fname)
            log.debug("Save model to {0}".format(model_fname))

            feature_fname = fname + "_feature.txt"
            io.save(self.embed, feature_fname)
            log.debug("Save feature to {0}".format(feature_fname))

            label_fname = fname + "_label.txt"
            io.save(self.lst_label, label_fname)
            log.debug("Save feature to {0}".format(label_fname))

        except Exception as e:
            log.error(e)
            return False
        return True

    def load_model(self, fname):
        try:
            model_fname = fname + "_model.txt"
            self.clf = io.load(model_fname)
            log.debug("Load model from {0}".format(model_fname))

            feature_fname = fname + "_feature.txt"
            self.embed = io.load(feature_fname)
            log.debug("Load features from {0}".format(feature_fname))

            label_fname = fname + "_label.txt"
            self.lst_label = io.load(label_fname)
            log.debug("Load features from {0}".format(label_fname))

        except Exception as e:
            log.error(e)
            return False
        return True

    def train(self, **kwargs):
        # 模型训练
        return self.fit(**kwargs)

    def predict(self, text):
        """
        预测样本
        :param text:
        :return:
        """
        if type(text) is str:
            text = [text]

        if self.embed_mode == "bow" or self.embed_mode == "tf-idf":
            embedded_x_valid = self.embed.transform(text)
        elif self.embed_mode == "w2v-avg":
            embedded_x_valid, _ \
                = embed.get_word2vec_embeddings(text, self.embed)
        else:
            embedded_x_valid = text
        y_predicted = self.clf.predict(embedded_x_valid)

        lst_tmp = []
        for i in range(0, len(y_predicted)):
            lst_tmp.append(self.lst_label[y_predicted[i]])
        y_predicted = lst_tmp
        y_predicted_prb = self.clf.predict_proba(embedded_x_valid)
        if len(y_predicted) == 1:
            y_predicted = y_predicted[0]
            y_predicted_prb = max(y_predicted_prb[0])

        return y_predicted, y_predicted_prb

    def evaluation(self, x, y, is_display=False):
        # 模型评估(新版本内容)，目前为了测试旧版本，暂时disable
        # =====
        lst_predicted_label, _ = self.predict(x)
        if type(lst_predicted_label) is not list:
            lst_predicted_label = [lst_predicted_label]

        lst_true_label = []
        for item in y:
            lst_true_label.append(self.lst_label[item])

        # 输出评估指标
        accuracy, precision, recall, f1, _ \
            = f1_score.sklearn_get_metrics(lst_true_label, lst_predicted_label)

        log.info("accuracy = %.3f, precision = %.3f,"
                 " recall = %.3f, f1 = %.3f"
                 % (accuracy, precision, recall, f1))

        for i in range(0, len(x)):
            if lst_true_label[i] != lst_predicted_label[i]:
                # 深度学习模型输入的x是已经编码化后的序列
                if type(x[i]) is np.ndarray:
                    text = self.tokenizer.sequences_to_texts([list(x[i])])[0]
                # 非深度学习模型输入的x是一个文本的原始形式
                else:
                    text = x[i]

                log.info(text + "\t" + lst_true_label[i] +
                         "\t" + lst_predicted_label[i])

        log.info("*" * 30)
        log.info("Model Precise: {0}".format(precision))
        matrix = metrics.confusion_matrix(lst_true_label, lst_predicted_label)
        log.info(matrix)
        report = metrics.precision_recall_fscore_support(lst_true_label,
                                                         lst_predicted_label)

        labels = sorted(set(lst_true_label))
        # # =====
        # # 基于统计学的方法
        # from sklearn.metrics import confusion_matrix
        # from vikinlp.ai_toolkit.inspection import statistics_based
        # import matplotlib.pyplot as plt
        # # cm = confusion_matrix(lst_true_label, lst_predicted_label)
        # plt.figure(figsize=(10, 10))
        # statistics_based.plot_confusion_matrix(matrix,
        #                                        classes=labels,
        #                                        normalize=True)
        # plt.show()
        # # =====

        class_precise = dict(zip(
            labels, map(lambda single_x: "%.2f" % round(single_x, 2),
                        report[0])))
        return {
            'class_precise': class_precise,
            'total_precise': precision
        }

    @abstractmethod
    def refresh_coef(self):
        pass

    @abstractmethod
    def fit(self):
        pass


class LogisticRegressionClassifier(QuestionClassfier):
    def __init__(self):
        QuestionClassfier.__init__(self)

        # 为了对sklearn中的fit函数进行封装，所以将clf指向self
        # self.clf = self
        self.clf = LogisticRegression(C=30.0, class_weight="balanced",
                                      solver='liblinear',
                                      n_jobs=-1, random_state=0)
        self.clf.predict_proba = self.clf.predict_proba
        self.original_fit = self.clf.fit
        self.clf.fit = self.fit
        self.coef_ = None

    # 需要参数如下：x,y,embed_path,max_num
    def fit(self, **kwargs):

        # 词典嵌入
        if self.embed_mode == "bow":
            embedded_x_train, self.embed = embed.bow(kwargs["x"])
        elif self.embed_mode == "tf-idf":
            embedded_x_train, self.embed = embed.tfidf(kwargs["x"])
        else:
            self.embed = embed.read_w2v(kwargs["embed_path"],
                                        kwargs["max_num"])
            embedded_x_train, _ = embed.get_word2vec_embeddings(kwargs["x"],
                                                                self.embed)

        # 模型训练
        # 注意不能省略赋值，否则会报错
        self.clf = self.original_fit(embedded_x_train, kwargs["y"])
        self.refresh_coef()

        # 输出模型评估报告
        return self.evaluation(self.x_valid, self.y_valid)

    def refresh_coef(self):
        self.coef_ = self.clf.coef_
