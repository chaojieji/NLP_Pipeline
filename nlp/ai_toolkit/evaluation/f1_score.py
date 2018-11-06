from sklearn.metrics import accuracy_score, f1_score,\
    precision_score, recall_score
from statsmodels.stats.proportion import proportion_confint


def calculate_confidence_int(correct_num, total_num):
    lower, upper = proportion_confint(correct_num, total_num, 0.05)
    return lower, upper


def sklearn_get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)

    # 计算置信区间，是一个很重要的评估指标
    correct_count = 0
    for i in range(0, len(y_test)):
        if y_test[i] == y_predicted[i]:
            correct_count += 1
        i += 1

    lower, upper = calculate_confidence_int(correct_count, len(y_test))
    confidence_interval = (lower, upper)
    return accuracy, precision, recall, f1, confidence_interval
