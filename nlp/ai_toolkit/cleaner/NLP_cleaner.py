import logging
from nlp.ai_toolkit.feeder import NLPFeeder
log = logging.getLogger(__name__)


def standardize_text(df, text_field):
    # Remove words that are not relevant
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    # Remove all irrelevant characters such as any non alphanumeric characters
    df[text_field]\
        = df[text_field].str.\
        replace(r"[A-Za-z0-9(),!?.@（）《》<>，！？“”。\'\`\"\_\n]", "")
    df[text_field] = df[text_field].str.replace("  ", "")
    return df


# 创建停用词list
def stop_words_list(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]
    return stopwords


# 对dataframe中的某一列某一行文本信息进行停止词剔除
def remove_single_stop_word(df, column_name, stopwords):
    out = ''
    original_text = df[column_name].split(" ")

    for word in original_text:
        if word not in stopwords:
            if word != '\t':
                out += word
                out += " "
    return out


# 对dataframe中的某一整列文本信息进行停止词剔除
def remove_all_stop_word(df, stop_words_path, column_name):
    stopwords = stop_words_list(stop_words_path)  # 这里加载停用词的路径
    df[column_name] = df.apply(remove_single_stop_word, axis=1,
                               column_name=column_name, stopwords=stopwords)
    return df


def explore_guangkai():
    # 读取数据源问题件
    data = NLPFeeder.read_file("../input/big_guangkai.txt", '@', '$',
                               ["意图", "问题"])
    log.info(data.describe())
    data.drop_duplicates(inplace=True)
    log.info(data.describe())

    data = standardize_text(data, "问题")
    log.info(data)
    data = remove_all_stop_word(data, "../input/stop_word.txt", "问题")
    log.info(data)


if __name__ == '__main__':
    explore_guangkai()
