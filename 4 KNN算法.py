# KNN 算法定义：如果一个样本和在特征空间中的k个最邻近的样本中的大多数属于一个类别，则该样本也属于这个类别
# k = 1, 容易受异常值影响


# 利用鸢尾花进行预测
# 1）获取数据
from sklearn.datasets import load_iris
iris = load_iris()
# 2) 数据集划分（划分测试和训练）
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=6)   #传特征值和目标值

# 3）特征工程：标准化(无量纲化处理)
from sklearn.preprocessing import StandardScaler
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)    # 训练集标准化
x_test = transfer.transform(x_test)          # 测试集标准化

# 4）KNN的预估器流程（训练）
from sklearn.neighbors import KNeighborsClassifier
estimator = KNeighborsClassifier(n_neighbors=3)
estimator.fit(x_train,y_train)

# 5）评估
# 方法1 ：直接比对真实值和预测值
y_predict = estimator.predict(x_test)
print(y_predict)
print("直接对比真实值和预测值:\n",y_test == y_predict)

# 方法2：计算准确率
score = estimator.score(x_test,y_test)
print("准确率为：\n",score)
