import copy
import logging

import pandas as pd
import matplotlib as plt

from nlp.ai_toolkit.util import sys
from nlp.ai_toolkit.preprocess import embed, transformer
from nlp.ai_toolkit.visualization import ml_visualization,\
    statistical_visualization
from nlp.ai_toolkit.cleaner import NLP_cleaner
from nlp.ai_toolkit.feeder import NLPFeeder
from nlp.ai_toolkit.inspection import model_based_w2v, model_based_not_w2v

log = logging.getLogger(__name__)


def dr_visualization(model, dimension, str_label_name,
                     w2v_num, feature_name, embed_file):
    list_label = transformer.construct_label(model.data,
                                             str_label_name)

    if model.embed_mode == "bow":
        bow = embed.construct_vector(model.data, embed.bow,
                                     feature_name=feature_name)
    elif model.embed_mode == "tf-idf":
        bow = embed.construct_vector(model.data, embed.tfidf,
                                     feature_name)
    else:
        vectors = embed.read_w2v(embed_file, w2v_num)
        bow = embed. \
            construct_vector(model.data,
                             embed.get_word2vec_embeddings, vectors,
                             feature_name)

    ml_visualization.pca(bow, list_label, dimension)


def des_visualization(model, lst_column_name):
    str_label_name = lst_column_name[0]
    str_feature_name = lst_column_name[1]

    log.info("原始数据描述：")
    log.info(model.data.describe())

    # 语料清洗
    log.info("数据清洗后描述：")
    data = NLP_cleaner.standardize_text(model.data, str_feature_name)
    # data = NLP_cleaner.remove_all_stop_word(data,
    #                                         "../input/stop_word.txt",
    #                                         "问题")
    log.info(data.describe())

    # 统计每个意图对应的问题数,升序排列并以柱状图来表示
    statistical_visualization.plot_label_distribution(data, str_label_name,
                                                      str_feature_name,
                                                      [str_label_name,
                                                       "Count"])

    # 每个问题词长度对应出现的意图数（相当于不同长度问题下的熵）
    data_word_number \
        = statistical_visualization.word_feature(copy.deepcopy(data),
                                                 str_feature_name)
    log.info("问题平均长度："
             + str(data_word_number[str_feature_name].mean()))
    statistical_visualization. \
        plot_label_distribution(data_word_number, str_feature_name,
                                str_label_name, ["Length of "
                                                 + str_feature_name,
                                                 "Count of "
                                                 + str_label_name])

    # 每个问题字长度对应出现的意图数（相当于不同长度问题下的熵）
    data_character_number \
        = statistical_visualization.character_feature(copy.deepcopy(data),
                                                      str_feature_name)
    statistical_visualization.plot_label_distribution(data_character_number,
                                                      str_feature_name,
                                                      str_label_name,
                                                      ["Length of "
                                                       + str_feature_name,
                                                       "Count of "
                                                       + str_label_name])

    # 统计词频
    statistical_visualization.get_frequency(data, str_feature_name)


def lime_visualization(model, feature, label, output_path,
                       max_sequence_length=10, tokenizer=None):

    label = model.lst_label.index(label)
    model_based_w2v \
        .visualize_one_exp(feature, label,
                           model.lst_label, model.clf, model.embed,
                           tokenizer, output_path=output_path,
                           max_sequence_length=max_sequence_length)


def word_importance(model, word_num):
    # 基于模型的方法
    # 只能用于BOW或者TF-IDF词典，不可用于Word2Vec
    model_based_not_w2v.batch_plot_importance(model.embed, model.clf,
                                              model.lst_label,
                                              word_num=word_num)


def get_data(str_input_path, lst_column_name,
             separator1, separator2):
    """
    载入数据
    :param str_input_path:
    :param lst_column_name:
    :param separator1:
    :param separator2:
    :return:
    """
    data = pd.DataFrame(columns=lst_column_name)

    list_file = sys.file_list(str_input_path)

    for item in list_file:
        cur_df = NLPFeeder.read_file(item,
                                     separator1, separator2,
                                     lst_column_name)
        data = pd.concat([data, cur_df], axis=0)

    data.drop_duplicates(inplace=True)
    return data


def plot_clr_curve(dic_lr_acc):
    lst_lr = []
    lst_acc = []
    i = 0
    for k, v in dic_lr_acc.items():
        lst_lr.append(k)
        lst_acc.append(v)
        i += 1

    plt.plot(lst_lr, lst_acc)
    plt.show()
