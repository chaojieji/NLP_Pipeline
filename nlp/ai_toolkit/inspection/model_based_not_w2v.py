import numpy as np
import matplotlib.pyplot as plt


# 获取模型中影响最大的特征（NLP中既为词或字），表现是方式为经过模型训练后，
# 这个特征前的比例系数较大
def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}

    # loop for each class
    classes = {}
    for class_index in range(model.coef_.shape[0]):
        word_importance\
            = [(el, index_to_word[i])
               for i, el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importance, key=lambda x: x[0],
                              reverse=True)
        tops = sorted(sorted_coeff[:n], key=lambda x: x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops': tops,
            'bottom': bottom
        }
    return classes


# 绘制label与关键特征之间的关系
def plot_important_words(top_scores, top_words, bottom_scores,
                         bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a, b) for a, b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])

    bottom_pairs = [(a, b) for a, b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)

    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]

    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]

    plt.figure(figsize=(10, 10))

    plt.subplot(121)
    plt.barh(y_pos, bottom_scores, align='center', alpha=0.5)
    plt.title('Bottom', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)

    plt.subplot(122)
    plt.barh(y_pos, top_scores, align='center', alpha=0.5)
    plt.title('Top', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)

    plt.subplots_adjust(wspace=0.8)
    plt.show()


def batch_plot_importance(model, clf, list_label, word_num=10):
    importance = get_most_important_features(model, clf, word_num)
    signal = ""
    while signal != "quit":
        class_name = input("Please intput class name:")
        index_num = list_label.index(class_name)
        top_scores = [a[0] for a in importance[index_num]['tops']]
        top_words = [a[1] for a in importance[index_num]['tops']]
        bottom_scores = [a[0] for a in importance[index_num]['bottom']]
        bottom_words = [a[1] for a in importance[index_num]['bottom']]
        plot_important_words(top_scores, top_words, bottom_scores,
                             bottom_words,
                             "Most important words for " + class_name)
