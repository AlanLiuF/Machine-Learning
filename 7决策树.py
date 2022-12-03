# 决策树
# 比如，我们举一个例子：
# 见图片决策树介绍（图）
# 因此决策树的思想：如何高效的进行学习
# 决策树的核心：如何鉴定特征的先后顺序
# 引入信息熵，信息增益的基础知识
'''
信息：消除随机不定性的东西
比如：小明今年18岁是信息，那么小明再过一年19岁，不是信息
信息的衡量 -- 信息量 -- 信息熵（单位是bit）
# 信息熵的公式以及信息熵的运用（图）
# 信息增益：整个集合的信息熵与给定特征条件下的条件信息熵之差，衡量不确定性减少的程度
# 看看哪个特征条件下的信息增益更大，哪个特征的顺序就最先
'''
# 用决策树对鸢尾花进行分类
# 1.获取数据集
from sklearn.datasets import load_iris
iris = load_iris()
# 2.划分数据集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(iris.data, iris.target, random_state=22) # 特征值，目标值，随机数种子
# 3.决策树预估器
from sklearn.tree import DecisionTreeClassifier
estimator = DecisionTreeClassifier(criterion="entropy")
# 4.模型评估
# 方法1 ：直接比对真实值和预测值
y_predict = estimator.predict(x_test)
print(y_predict)
print("直接对比真实值和预测值:\n",y_test == y_predict)

# 方法2：计算准确率
score = estimator.score(x_test,y_test)
print("准确率为：\n",score)
