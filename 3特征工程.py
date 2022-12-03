# 数据和特征工程决定了机器学习的上线，而模型和算法只是去逼近这个上线
# 分为特征抽取，特征预处理，特征降维

# 一.特征抽取
# 将文本/图像转换成可以用于机器学习的数字数据
# 1.字典的特征抽取：将类别通过哑变量转换成矩阵
from sklearn.feature_extraction import DictVectorizer
dict = [{'city':"北京",'temperature':100},{'city':"上海",'temperature':60},{'city':"深圳",'temperature':30}]
transfer = DictVectorizer(sparse=False)               # 实例化一个转换器类
data_new = transfer.fit_transform(dict)
print(data_new)


# 2.文本的特征抽取：单词作为特征（特征词）
# 方法1：countvectorizer
data = ['life is short,i like like python','life is long,i dislike python']
from sklearn.feature_extraction.text import CountVectorizer
transfer = CountVectorizer()                           # 实例化一个转换器类
data_new2 = transfer.fit_transform(data)
print(transfer.get_feature_names())
print(data_new2.toarray())
# 返回的结果是统计特征值出现的个数，而不是是否出现
# 如果文本是中文，怎么办？ 需要手动分词，不然识别的是句子，不是单词
dataa = ['我 爱 北京 天安门','天安门 上 太阳 升']
from sklearn.feature_extraction.text import CountVectorizer
transfer = CountVectorizer()                           # 实例化一个转换器类
data_new3 = transfer.fit_transform(dataa)
print(transfer.get_feature_names())
print(data_new3.toarray())
# 注意，可以在CountVectorizer()括号中加上停用词，stop_words = ['is','too']

# 手动拆开不实用啊，所以让机器自动帮我们分词：
import jieba
a = ['但愿天下人都能以三生的相约来看待，不辜负前世，不枉费来生。','最重要的此生此世，则能以爱为灯，穿越时空','以美为光，照亮生命。']
new = []
for sent in a:
    new.append(' '.join(list(jieba.cut(sent))))

from sklearn.feature_extraction.text import CountVectorizer
transfer = CountVectorizer()                           # 实例化一个转换器类
new2 = transfer.fit_transform(new)
print(transfer.get_feature_names())
print(new2.toarray())
# ['三生', '以美为', '但愿', '前世', '天下人', '枉费', '此世', '此生', '照亮', '生命', '相约', '看待', '穿越时空', '辜负', '重要']
# [[1 0 1 1 1 1 0 0 0 0 1 1 0 1 0]
#  [0 0 0 0 0 0 1 1 0 0 0 0 1 0 1]
#  [0 1 0 0 0 0 0 0 1 1 0 0 0 0 0]]

# 方法2：TfidfVectorizer:衡量一个词在文章的重要程度
# 关键词：在某一个类别的文章，出现次数很多，但在其他类别的文章中，出现次数少
# TF 词频
# IDF 逆向文本频率：总文件数除以出现这个词的文件数，再取一个以10为底的log
# tfidf = TF * IDF
# 代码写法与上一个类似



# 二.特征预处理
# 无量纲化处理：归一化/标准化
# 归一化：将数据影射到0-1
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
data1 = pd.read_excel(r"C:\Users\lfz00\Desktop\LFZ\PYTHON\data\归一化.xlsx")
print(data1)
transfer = MinMaxScaler()
data1_new = transfer.fit_transform(data1)
print(data1_new)
# 但是，归一化受最大最小值的影响较大，万一这两个值是异常的，怎么办？

# 于是：标准化
from sklearn.preprocessing import StandardScaler
data2 = pd.read_excel(r"C:\Users\lfz00\Desktop\LFZ\PYTHON\data\归一化.xlsx")
transfer = StandardScaler()
data2_new = transfer.fit_transform(data2)
print(data2_new)



# 三.特征降维：减少随机变量（特征）的个数，得到一组不相关的主变量的过程

# 1：特征选择
# 【1】.Filter 过滤低方差特征
# (1) 方差选择法
from sklearn.feature_selection import VarianceThreshold
data3 =pd.read_excel(r"...")
transfer = VarianceThreshold(threshold = 10)
data3_new = transfer.fit_transform(data3)
print(data3_new)
# (2) 相关系数
# 用scipy
# 【2】.Embedded
# 决策树，正则化，深度学习


# 2：主成分分析
# 尽可能降低原有数据的维数
# 如果传小数，表示保留百分之几的信息；如果整数，表示要保留多少特征
from sklearn.decomposition import PCA
data4 = [[2,8,4,5],[1,5,3,8],[1,6,9,2]]
transfer = PCA(n_components = 2)      # 把4个特征转换成两个特征
data4_new = transfer.fit_transform(data4)
print(data4_new)




