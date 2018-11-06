import pickle
import copy
import logging

from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from nlp.ai_toolkit.preprocess import transformer

log = logging.getLogger(__name__)


def bow(list_text):
    cv = CountVectorizer()
    emb = cv.fit_transform(list_text)
    return emb, cv


def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    emb = tfidf_vectorizer.fit_transform(data)
    return emb, tfidf_vectorizer


# 如果path_solidified参数被指定了，
# 那么就说明当前读入的这个词向量文件是一个之前已经通过pickle固化了的词典
# read top n word vectors, i.e. top is 10000
def read_w2v(path, topn, path_solidified=None):
    lines_num, dim = 0, 0
    word_vector = {}
    iw = []
    wi = {}
    log.info("Loading Vocabulary ...")
    if path_solidified is None:
        with open(path, encoding='utf-8', errors='ignore') as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                    continue
                lines_num += 1
                tokens = line.rstrip().split(' ')
                word_vector[tokens[0]]\
                    = np.asarray([float(x) for x in tokens[1:]])
                iw.append(tokens[0])
                if topn != 0 and lines_num >= topn:
                    break
        for i, w in enumerate(iw):
            wi[w] = i

    else:
        with open(path_solidified, "rb") as f:
            word_vector = pickle.load(f)
    log.info("Loaded Vocabulary ...")
    return word_vector


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k)
                      for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k)
                      for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(list_text, vocab):
    embeddings = []
    for item in list_text:
        embeddings.append(get_average_word2vec(item, vocab,
                                               generate_missing=False))
    # 返回的第二个参数没有实际意义，这是为了在使用装饰器的时候能够保证输出格式一致
    return embeddings, None


def sequence_tokenize(list_label, x_train, y_train, x_valid,
                      y_valid, x_test, y_test, embed_path, max_vocab):
    vocab = read_w2v(embed_path, max_vocab)
    embedding_dim = len(vocab[list(vocab.keys())[0]])
    vocab_size = len(vocab)
    tokenizer = Tokenizer(num_words=vocab_size)

    clean_questions = DataFrame({"Column1": x_train + x_valid + x_test,
                                 "Column2": y_train + y_valid + y_test})

    # tokenizer的作用就是为一段文本建立索引唯一的字典
    tokenizer.fit_on_texts(clean_questions["Column1"].tolist())

    word_index = tokenizer.word_index
    log.info('Found %s unique tokens.' % len(word_index))

    # ====
    # 词典的确立
    # 建立词和对应的词向量
    embedding_weights = np.zeros((len(word_index)+1, embedding_dim))
    for word, index in word_index.items():
        embedding_weights[index, :]\
            = vocab[word] if word in vocab else np.random.rand(embedding_dim)

    # ====
    # label要以统一的[文本-数字编号]对来进行切换
    digital_label = transformer.construct_label(clean_questions,
                                                "Column2")
    # 将文本类别名转换为数字类别名
    clean_questions["Column2"] = digital_label

    list_sample = []
    for item in [[x_train, y_train], [x_valid, y_valid], [x_test, y_test]]:
        original_y = copy.deepcopy(item[1])
        # 将文本类别名转换为数字类别名
        item[1] = to_categorical(np.asarray(item[1]))
        list_sample.append([item[0], item[1], original_y])

    return embedding_weights, list_sample[0][0], list_sample[0][1], \
        list_sample[1][0], list_sample[1][1], list_sample[2][0], \
        list_sample[2][1], embedding_dim, list_label, y_train, y_valid,\
        y_test, tokenizer, list_sample[0][2], list_sample[1][2],\
        list_sample[2][2]


def aggregate_text(func):
    def wrapper(*args):
        list_sequence = []
        args[0].apply(func, axis=1, column_name=args[1],
                      list_sequence=list_sequence)
        return list_sequence
    return wrapper


@aggregate_text
def aggregate_single_text(df, column_name, list_sequence):
    for item in df[column_name].split(" "):
        if item is not "":
            list_sequence.append(item)


@aggregate_text
def aggregate_single_text(df, column_name, list_sequence):
    list_sequence.append(df[column_name])


def construct_vector(data, fun, vocabulary=None, feature_name=""):
    list_sequence = aggregate_single_text(data, feature_name)

    if vocabulary is None:
        cv_emb, cv_model = fun(list_sequence)
    else:
        cv_emb, cv_model = fun(list_sequence, vocabulary)

    if type(cv_emb) is list:
        cv_emb = np.array(cv_emb)
    else:
        cv_emb = cv_emb.toarray()

    return cv_emb


if __name__ == '__main__':
    w2v = read_w2v("/home/jichaojie/Bitmain/VikiNLU/data/"
                   "sgns.baidubaike.bigram-char",
                   200)
    log.info(w2v["们"])
