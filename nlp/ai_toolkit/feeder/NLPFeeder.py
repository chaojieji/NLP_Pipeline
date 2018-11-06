import logging

import pandas as pd
import xlrd
import jieba

from nlp.ai_toolkit.util.sys import file_list
log = logging.getLogger(__name__)


def read_file(input_path, separator1, separator2, column_name):
    df = pd.read_csv(input_path, separator1, names=column_name)
    df.replace('[' + separator2 + ']', '', regex=True, inplace=True)
    return df


def read_xls(input_path, output_path, str_separator):
    excel_file = xlrd.open_workbook(input_path)
    log.info(excel_file.sheet_names())
    sheet = excel_file.sheet_by_index(0)
    log.info(sheet.name, sheet.nrows, sheet.ncols)
    label_name = ""
    for i in range(1, sheet.nrows):
        row = sheet.row_values(i)

        if row[0] != "":
            if row[1] != "":
                label_name = row[1]
            context = row[0]

            # 通过结巴分词的全模式对文本内容进行分词
            acc_context = ' '.join(jieba.cut(context, cut_all=False))

            with open(output_path + label_name + ".txt", 'a') as f:
                f.write(label_name + str_separator + acc_context + "\n")


def read_xls_by_sheet(input_path, output_path, str_separator):
    excel_file = xlrd.open_workbook(input_path)
    sheet_name = excel_file.sheet_names()

    for single_sheet in sheet_name:
        sheet = excel_file.sheet_by_name(single_sheet)

        log.info(sheet.name, sheet.nrows, sheet.ncols)

        for i in range(1, sheet.nrows):
            row = sheet.row_values(i)
            context = row[0].strip()
            # 通过结巴分词的全模式对文本内容进行分词
            acc_context = ' '.join(jieba.cut(context, cut_all=False))
            with open(output_path + single_sheet + "_non_rule.txt", 'a') as f:
                f.write(single_sheet + str_separator + acc_context + "\n")


def add_prefix(input_path, output_path, str_prefix, str_separator):
    with open(input_path, 'r') as f:
        lst_text = f.readlines()

    with open(output_path, 'w') as f:
        for item in lst_text:
            acc_context = ' '.join(jieba.cut(item, cut_all=False))
            f.write(str_prefix + str_separator + acc_context)


def word2char(input_path, output_path, str_separator):
    """
    将以单词为基础的语料，改为以字为基础的
    :return:
    """

    lst_file = file_list(input_path)

    for file_name in lst_file:

        with open(file_name, 'r') as f:
            lst_text = f.readlines()

        path_postfix = file_name.split("/")[-1]
        with open(output_path + "/" + path_postfix, 'w') as f:
            for item in lst_text:

                sample = item.split(str_separator)

                text_tmp = sample[1].replace(" ", "")
                lst_char = []
                for element in text_tmp:
                    lst_char.append(element)
                f.write(sample[0] + str_separator + " ".join(lst_char).strip()
                        + "\n")


if __name__ == '__main__':
    output_dir = "../../data/query_binary_addition/"
    # remove_folder(output_dir)
    # read_xls("../../data/original_xls/XiaodouLuo_query.xlsx",
    #          output_dir, "$@")
    # read_xls_by_sheet("../../data/original_xls/意图数据_20181013.xlsx",
    #                   output_dir, "$@")

    # add_prefix("/home/jichaojie/Downloads/train_chat.txt",
    #            "/home/jichaojie/Downloads/train_chat_labeled.txt",
    #            "chat", "$@")

    word2char("/home/jichaojie/Bitmain/VikiNLU/data/query_origin",
              "/home/jichaojie/Bitmain/VikiNLU/data/query_origin_char", "$@")

    pass
