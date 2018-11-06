import matplotlib as plt


# 需要手动下载simhei字体，拷贝到matplotlib中的字库目录下
# 参考https://www.jb51.net/article/115533.htm
def set_chinese_font():
    plt.rcParams[u'font.sans-serif'] = ['simhei']
    plt.rcParams['axes.unicode_minus'] = False
