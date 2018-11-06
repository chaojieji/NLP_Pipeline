import random
from collections import defaultdict
import logging

from lime.lime_text import LimeTextExplainer
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

from nlp.ai_toolkit.preprocess import embed
from nlp.ai_toolkit.inspection import model_based_not_w2v

log = logging.getLogger(__name__)

word2vec = None
clf = None
g_tokenizer = None
MAX_SEQUENCE_LENGTH = 0


def explain_one_instance(instance, class_names, top_label, classifier,
                         vocab, max_sequence_length):
    global g_tokenizer
    global MAX_SEQUENCE_LENGTH

    # 深度学习的方法不需要载入词典，而非深度学习的方法（sklearn中的方法）需要在这里替换为词典
    if g_tokenizer is None:
        # 非深度学习模型传进来的是"在 哪里"
        vectorized_example = embed.get_average_word2vec(instance, vocab,
                                                        generate_missing=False)
        vectorized_example = [vectorized_example]
    else:
        sequences = g_tokenizer.texts_to_sequences([instance])
        # 补齐长度,最终用于输入模型的文本
        vectorized_example\
            = pad_sequences(sequences, maxlen=max_sequence_length)
        MAX_SEQUENCE_LENGTH = max_sequence_length

    predicated_class_name = classifier.predict(vectorized_example)[0]
    # a = class_names
    log.info("Predicated Class: " + predicated_class_name)

    # 如果模型预测错误，需要分析为什么会预测到这个标签
    predicated_class = class_names.index(predicated_class_name)

    explainer = LimeTextExplainer(class_names=class_names)
    # predicated_class： 真正预测到的标签
    # top_label: 实际标记的标签
    exp = explainer.explain_instance(instance, word2vec_pipeline,
                                     num_features=6,
                                     labels=(predicated_class, top_label,))
    return exp


def visualize_one_exp(feature, label, class_names, classifier, vocab,
                      tokenizer=None, output_path=None,
                      max_sequence_length=10):
    # log.info('Index: %d' % index)
    # log.info('Context: '+ str(features[index]))
    # 一定要做这一步，因为word2vec_pipeline需要这两个变量，但是又没有办法直接传入函数
    global word2vec, clf, g_tokenizer
    word2vec = vocab
    clf = classifier
    g_tokenizer = tokenizer

    exp = explain_one_instance(feature, class_names,
                               label,
                               classifier, vocab, max_sequence_length)

    log.info('Labeled Class: %s' % class_names[label])
    if output_path is not None:
        exp.save_to_file(output_path)


def word2vec_pipeline(examples):
    global word2vec
    global clf
    global MAX_SEQUENCE_LENGTH

    tokenized_list = []

    if g_tokenizer is None:
        for example in examples:
            vectorized_example\
                = embed.get_average_word2vec(example, word2vec,
                                             generate_missing=False)
            tokenized_list.append(vectorized_example)
    else:

        sequences = g_tokenizer.texts_to_sequences(examples)
        # 补齐长度,最终用于输入模型的文本
        tokenized_list = pad_sequences(sequences,
                                       maxlen=MAX_SEQUENCE_LENGTH)

    tmp = clf.predict_proba(tokenized_list)
    return tmp


# 以下是批量处理方法
def get_statistical_explanation(test_set, sample_size, fun, vocab, classifier):

    # 一定要做这一步，因为fun需要这两个变量，但是又没有办法直接传入函数
    global word2vec, clf
    word2vec = vocab
    clf = classifier
    log.info(len(test_set))
    sample_sentences = random.sample(test_set, sample_size)
    explainer = LimeTextExplainer()

    labels_to_sentences = defaultdict(list)
    contributors = defaultdict(dict)

    # First, find contributing words to each class
    for sentence in sample_sentences:
        probabilities = fun([sentence])
        curr_label = probabilities[0].argmax()
        labels_to_sentences[curr_label].append(sentence)
        instance = sentence

        exp = explainer.explain_instance(instance, fun,
                                         num_features=20, labels=[curr_label])
        listed_explanation = exp.as_list(label=curr_label)

        for word, contributing_weight in listed_explanation:
            if word in contributors[curr_label]:
                contributors[curr_label][word].append(contributing_weight)
            else:
                contributors[curr_label][word] = [contributing_weight]

    log.info(contributors[1])
    log.info("=====")
    average_contributions = {}
    sorted_contributions = {}
    for label, lexica in contributors.items():
        curr_label = label
        curr_lexica = lexica
        average_contributions[curr_label] = pd.Series(index=curr_lexica.keys())
        for word, scores in curr_lexica.items():
            average_contributions[curr_label].loc[word]\
                = np.sum(np.array(scores))/sample_size
        detractors = average_contributions[curr_label].sort_values()
        supporters\
            = average_contributions[curr_label].sort_values(ascending=False)
        sorted_contributions[curr_label] = {
            'detractors': detractors,
            'supporters': supporters
        }
    return sorted_contributions


def plot_statistical_explanation(test_set, sample_size, fun,
                                 label_list, vocab, classifier):
    sorted_contributions\
        = get_statistical_explanation(test_set, sample_size,
                                      fun, vocab, classifier)

    label_dict = {}
    i = 0
    for item in label_list:
        label_dict[item] = i
        i += 1

    log.info(label_list)
    log.info(sorted_contributions)
    signal = ""
    while signal != "quit":
        class_name_text = input("Please intput class name:")
        class_name = label_dict[class_name_text]
        top_words\
            = sorted_contributions[class_name]['supporters'][:10]\
            .index.tolist()
        top_scores\
            = sorted_contributions[class_name]['supporters'][:10].tolist()
        bottom_words\
            = sorted_contributions[class_name]['detractors'][:10]\
            .index.tolist()
        bottom_scores\
            = sorted_contributions[class_name]['detractors'][:10].tolist()

        model_based_not_w2v.plot_important_words(top_scores, top_words,
                                                 bottom_scores, bottom_words,
                                                 "Most important words for "
                                                 + str(class_name_text))
        signal = input()
