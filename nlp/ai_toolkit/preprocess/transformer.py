def map_label_id(df, column_name, dict_label):
    df[column_name] = dict_label[df[column_name]]
    return df[column_name]


def get_distinct_label(df, column_name):
    df_label = df.drop_duplicates(subset=[column_name])
    list_label = list(df_label[column_name])
    return list_label


def construct_label(df, column_name):
    list_label = get_distinct_label(df, column_name)
    dict_label = {}
    index = 0
    for item in list_label:
        dict_label[item] = index
        index += 1

    df[column_name] = df.apply(map_label_id, axis=1,
                               column_name=column_name, dict_label=dict_label)
    return list(df[column_name])


def label_distribution(df, main_column, auxiliary_column):
    df_grouped = df.groupby([main_column], sort=False)[auxiliary_column] \
        .nunique().sort_values(ascending=True)
    # 将那些对应样本数<2的标签显性返回，因为这部分样本是没有办法划分为三个集合的
    # 还有种情况是样本的采样率太低样本太少，导致按照那个采样率采会导致一个集合中一个样本都没有
    return df_grouped[df_grouped < 2], df_grouped
