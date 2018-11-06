#!/usr/bin/env python
# encoding: utf-8


import re
import collections
import os
from nlp.util.langconv import Converter

PROJECT_DIR = os.path.realpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../../'))
SYSTEM_DIR = os.path.realpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../../../'))
CORPUS_DIR = os.path.join(PROJECT_DIR, 'data', 'corpus')
SEP_MARK = '$@'
g_word2pinyin = {}


def simple2tradition(line):
    """
    line: utf8 string
    """
    # 将简体转换成繁体
    line = Converter('zh-hant').convert(line.decode('utf-8'))
    line = line.encode('utf-8')
    return line


def tradition2simple(line):
    """
    line: utf8 string
    """
    # 将繁体转换成简体
    line = Converter('zh-hans').convert(line.decode('utf-8'))
    line = line.encode('utf-8')
    return line


def search_num(text):
    return re.compile(r"[+-]?\d+(?:\.\d+)?")


def is_chinese(hanzi):
    return hanzi >= u'\u4e00' and hanzi <= u'\u9fff'


def is_punctuation(c):
    c = unicode(c)
    hanzi = u"！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
    en = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    return c in hanzi or c in en


def convert(data):
    """ convert unicode or dict with unicode string
    to dict with string"""
    if isinstance(data, basestring):
        return data.encode('utf8')
    elif isinstance(data, collections.Mapping):
        rst = {}
        for key, value in map(convert, data.iteritems()):
            rst[key] = value
        return rst
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    else:
        return data


def pinyin_of_hanzi(aword):
    """ single chinese word to pinyin.

    return ['xxx', 'yyy']

    >>> pinyin_of_hanzi(u'你')
    """
    assert(is_chinese(aword))
    global g_word2pinyin
    if g_word2pinyin:
        try:
            return g_word2pinyin[aword]
        except KeyError:
            return []

    path = os.path.join(CORPUS_DIR, 'hanzi2pinyin.txt')
    lines = [line.rstrip('\r\n') for line in open(path)]
    for line in lines:
        word, pinyins = tuple(line.split('\t'))
        pinyins = pinyins.lower()
        pinyins = pinyins.split(' ')
        g_word2pinyin[word.decode('utf8')] = pinyins
    return g_word2pinyin[aword]


def pinyin_of_chinese(words):
    """
    >>> pinyin_of_chinese(u'账号变更')
    """
    alternative_pinyins = []
    for w in list(words):
        if not is_chinese(w):
            continue
        if not alternative_pinyins:
            alternative_pinyins = pinyin_of_hanzi(w)
            continue
        cur_pinyins = pinyin_of_hanzi(w)
        if cur_pinyins:
            new_pinyins = []
            for pre_pinyin in alternative_pinyins:
                for pinyin in cur_pinyins:
                    new_pinyins.append(pre_pinyin + ' ' + pinyin)
            alternative_pinyins = new_pinyins
    return alternative_pinyins


dict ={u'零': 0, u'一': 1, u'二': 2, u'俩': 2, u'两': 2, u'三': 3, u'四': 4, u'五': 5, u'六': 6, u' 七': 7, u'八': 8, u'九': 9, u'十': 10, u'百': 100, u'千': 1000, u'万': 10000, u'０': 0, u'１ ': 1, u'２': 2, u'３': 3, u'４': 4, u'５': 5, u'６': 6, u'７': 7, u'８': 8, u'９': 9, u'壹': 1, u'贰': 2, u'叁': 3, u'肆': 4, u'伍': 5, u'陆': 6, u'柒': 7, u'捌': 8, u'玖': 9, u'拾': 10, u'佰': 100, u'仟': 1000, u'萬': 10000, u'亿': 100000000}


def cn2digit(a, encoding="utf-8"):
    m = re.search(r"(\d)*", a)
    if m and m.group() == a:
        return a
    if isinstance(a, str):
        a = a.decode(encoding)
    count = 0
    result = 0
    tmp = 0
    Billion = 0
    while count < len(a):
        tmpChr = a[count]
        tmpNum = dict.get(tmpChr, None)
        # 如果等于1亿
        if tmpNum == 100000000:
            result = result + tmp
            result = result * tmpNum
            # 获得亿以上的数量，将其保存在中间变量Billion中并清空result
            Billion = Billion * 100000000 + result
            result = 0
            tmp = 0
        # 如果等于1万
        elif tmpNum == 10000:
            result = result + tmp
            result = result * tmpNum
            tmp = 0
        # 如果等于十或者百，千
        elif tmpNum >= 10:
            if tmp == 0:
                tmp = 1
            result = result + tmpNum * tmp
            tmp = 0
        # 如果是个位数
        elif tmpNum is not None:
            tmp = tmp * 10 + tmpNum
        count += 1
    result = result + tmp
    result = result + Billion
    return result


if __name__ == '__main__':
    print(cn2digit("一千三百"))
    print(cn2digit("零点三"))  # 不支持
