import copy
import logging

import matplotlib.pyplot as plt
import nltk
from pandas.core.frame import DataFrame

from nlp.ai_toolkit.visualization import util
from nlp.ai_toolkit.util import zh
from nlp.ai_toolkit.feeder import NLPFeeder
from nlp.ai_toolkit.cleaner import NLP_cleaner
from nlp.ai_toolkit.preprocess import embed, transformer

log = logging.getLogger(__name__)


def plot_label_distribution(df, main_column, auxiliary_column, axis_label):
    transformer.label_distribution()

    _, df_grouped = transformer.label_distribution(df, main_column,
                                                   auxiliary_column)

    df_grouped.plot(kind='bar')
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.show()


def count_word(df, column_name):
    word_number = df[column_name].count(" ") + 1
    return word_number


def count_character(df, column_name):
    character_number = zh.count_characters(df[column_name].strip())
    return character_number


def word_feature(df, column_name):
    df[column_name] = df.apply(count_word, axis=1, column_name=column_name)
    return df


def character_feature(df, column_name):
    df[column_name] = df.apply(count_character, axis=1,
                               column_name=column_name)
    return df


def get_frequency(df, column_name):
    word_sequence = embed.aggregate_single_text(df, column_name)
    fd = nltk.FreqDist(word_sequence)
    item = fd.items()
    # log.info ' '.join(keys)
    dicts = dict(item)
    sort_dict = sorted(dicts.items(), key=lambda d: d[1], reverse=True)

    log.info("共出现不同词汇个数" + str(len(sort_dict)))
    log.info("所有词汇词频:\n" + str(sort_dict))


def explore_guangkai():
    util.set_chinese_font()
    # 读取数据源问题件
    data = NLPFeeder.read_file("../input/big_guangkai.txt", '@', '$',
                               ["意图", "问题"])
    log.info("原始数据描述：")
    log.info(data.describe())
    data.drop_duplicates(inplace=True)
    log.info("去重后数据描述：")
    log.info(data.describe())

    # 统计每个意图对应的问题数,升序排列并以柱状图来表示
    plot_label_distribution(data, "意图", "问题", ["意图", "样本数"])
    #
    # # 每个问题词长度对应出现的意图数（相当于不同长度问题下的熵）
    data_word_number = word_feature(copy.deepcopy(data), "问题")
    plot_label_distribution(data_word_number, "问题", "意图", ["问题中词长度", "意图数"])
    #
    # # 每个问题字长度对应出现的意图数（相当于不同长度问题下的熵）
    data_character_number = character_feature(copy.deepcopy(data), "问题")
    plot_label_distribution(data_character_number, "问题", "意图", ["问题中字长度", "意图数"])

    # 统计词频
    get_frequency(data, "问题")
    data = NLP_cleaner.remove_all_stop_word(data, "../input/stop_word.txt",
                                            "问题")
    get_frequency(data, "问题")


def validate_distribution(list_dataset, feature_name, label_name,
                          is_display=True):
    # 验证生成数据的标签同比例特性
    i = 1
    for item in list_dataset:
        if is_display:
            plt.subplot(220+i)
        i += 1
        combination = {label_name: item[0], feature_name: item[1]}
        data_visualization = DataFrame(combination)
        plot_label_distribution(data_visualization,
                                                          label_name, feature_name,
                                                          [label_name, "Count"])


if __name__ == '__main__':
    explore_guangkai()
