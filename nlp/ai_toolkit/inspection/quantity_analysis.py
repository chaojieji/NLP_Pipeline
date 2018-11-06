import math
import copy

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np

from nlp.classifier import lstm


# 每个类别下的样本数量必须至少大于10
def evaluate_scale(x, split_num):
    kf = KFold(n_splits=split_num)

    lst_train = []
    lst_test = []
    for train_index, test_index in kf.split(x):
        lst_train.append(train_index)
        lst_test.append(test_index)

    return lst_train, lst_test


def plot_scale_acc_curve(dic_data):
    lst_index = []
    lst_low = []
    lst_high = []
    subclass_num\
        = math.ceil(math.sqrt(len(dic_data[list(dic_data.keys())[0]][
                                      "class_confidence_interval"].keys())
                              + 1))

    for item in dic_data:
        lst_index.append(item)
        lst_low.append(dic_data[item]["confidence_interval"][0])
        lst_high.append(dic_data[item]["confidence_interval"][1])
    plt.subplot(subclass_num, subclass_num, 1)
    plt.title("Total")
    plt.plot(lst_index, lst_low)
    plt.plot(lst_index, lst_high)
    plt.legend(['low acc', 'high acc'])
    plt.tick_params(labelsize=6)
    plt.subplots_adjust(wspace=0.6, hspace=0.6)
    acc_index = 2
    for item in dic_data[list(
            dic_data.keys())[0]]["class_confidence_interval"]:
        lst_index = []
        lst_low = []
        lst_high = []
        for key_name in dic_data.keys():
            lst_index.append(key_name)
            lst_high.append(dic_data[key_name][
                                "class_confidence_interval"][item][1])
            lst_low.append(dic_data[key_name][
                               "class_confidence_interval"][item][0])

        plt.subplot(subclass_num, subclass_num, acc_index)
        acc_index += 1

        str_tmp = ""
        for key_name in dic_data.keys():
            str_tmp += "-" + str(dic_data[key_name]["class_num"][item])

        plt.title(str(item) + str_tmp)
        plt.plot(lst_index, lst_low)
        plt.plot(lst_index, lst_high)
        plt.legend(['low acc', 'high acc'])
        plt.tick_params(labelsize=6)
        plt.subplots_adjust(wspace=0.6, hspace=0.6)
    plt.show()


def all_np(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result


def generate_scale_acc_report(folds_performance, fold_num, overall_performance,
                              total_sample_num):
    fold_per = {}
    fold_per["confidence_interval"] = (0, 0)
    fold_per["class_confidence_interval"] = {}
    fold_per["class_num"] = {}
    for i in range(0, len(folds_performance)):
        # if "total_precise" is not in tmp.keys():
        #     tmp["total_precise"] = 0

        fold_per["confidence_interval"] \
            = tuple([a+b for a, b in zip(fold_per["confidence_interval"],
                                         folds_performance[i][
                                             "confidence_interval"])])

        for item in folds_performance[i]["class_confidence_interval"].keys():
            if item not in fold_per["class_confidence_interval"].keys():
                fold_per["class_confidence_interval"][item] = (0, 0)

            fold_per["class_confidence_interval"][item] \
                = tuple([a+b
                         for a, b in
                         zip(fold_per["class_confidence_interval"][item],
                             folds_performance[i][
                                 "class_confidence_interval"][item])])

            # 类别下样本数
            if item not in fold_per["class_num"].keys():
                fold_per["class_num"][item] = 0
            fold_per["class_num"][item]\
                += folds_performance[i]["class_num"][item]

    fold_per["confidence_interval"] = (fold_per["confidence_interval"][0]
                                       / float(fold_num),
                                       fold_per["confidence_interval"][1]
                                       / float(fold_num))

    for item in fold_per["class_confidence_interval"]:
        fold_per["class_confidence_interval"][item]\
            = (fold_per["class_confidence_interval"][item][0] /
               float(fold_num),
               fold_per["class_confidence_interval"][item][1] /
               float(fold_num))
        fold_per["class_num"][item] = fold_per[
                                          "class_num"][item] / float(fold_num)
    overall_performance[total_sample_num] = fold_per


def extend_report(predictor, incomplete_report, fold_report):
    sample_num_class = all_np(predictor.y_train_origin
                              + predictor.y_test_origin)
    sample_num_class_caption = {}
    for item in sample_num_class:
        sample_num_class_caption[predictor.lst_label[item]]\
            = sample_num_class[item]
    incomplete_report['class_num'] = sample_num_class_caption
    fold_report.append(incomplete_report)


def load_fold_data(fold_num, lst_train_index, lst_valid_index, x_data_dump,
                   x_data_origin_dump, y_data_dump,
                   y_data_origin_dump, predictor):
    x_train = []
    x_train_origin = []
    y_train = []
    y_train_origin = []

    for item in lst_train_index[fold_num]:
        x_train.append(x_data_dump[item])
        x_train_origin.append(x_data_origin_dump[item])
        y_train.append(y_data_dump[item])
        y_train_origin.append(y_data_origin_dump[item])

    x_valid = []
    y_valid = []
    y_valid_origin = []

    for item in lst_valid_index[fold_num]:
        x_valid.append(x_data_dump[item])
        y_valid.append(y_data_dump[item])
        y_valid_origin.append(y_data_origin_dump[item])

    predictor.x_train = np.array(x_train)
    predictor.y_train = np.array(y_train)
    predictor.x_train_origin = x_train_origin
    predictor.y_train_origin = y_train_origin
    predictor.x_valid = np.array(x_valid)
    predictor.y_valid = np.array(y_valid)
    predictor.y_valid_origin = y_valid_origin


def bootstrap_sample(model, lst_range, fold_num, max_iter_num,
                     max_sequence, max_vob):
    single_model = model.Instance(lstm.lstm)

    predictor, parameters = single_model.model_selection()
    parameters.verbose = False

    overall_per = {}
    for increment in lst_range:

        total_sample_num = single_model.data_cv_split(predictor, increment, 0)

        # 对这批数据进行KFold操作
        # 注意，只有x_train的数据才是我们要的,x_test为废弃的数据
        lst_train, lst_valid \
            = evaluate_scale(predictor.x_train, fold_num)

        single_model.data_transformer(predictor, max_sequence, max_vob)

        # 备份这批数据
        x_data_dump = copy.deepcopy(predictor.x_train)
        x_data_origin_dump = copy.deepcopy(predictor.x_train_origin)
        y_data_dump = copy.deepcopy(predictor.y_train)
        y_data_origin_dump = copy.deepcopy(predictor.y_train_origin)

        lst_fold_performance = []
        for i in range(0, len(lst_train)):
            load_fold_data(i, lst_train, lst_valid, x_data_dump,
                           x_data_origin_dump, y_data_dump, y_data_origin_dump,
                           predictor)

            performance = single_model.train_model(predictor, parameters,
                                                   max_iter_num)
            extend_report(predictor, performance, lst_fold_performance)
        generate_scale_acc_report(lst_fold_performance, fold_num, overall_per,
                                  total_sample_num)

    # 绘制曲线
    plot_scale_acc_curve(overall_per)


if __name__ == "__main__":
    bootstrap_sample(lstm, [0.1, 0.25], 2, 2, 20, 1000)
