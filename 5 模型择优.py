# 一. 交叉验证：为了让模型更加准确可信
# 训练集：训练集+验证集

# 二. 网格搜索：关于怎么得到最好的K值（KNN算法）
# 鸢尾花案例增加K值调优
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

# 加入网格搜索和交叉验证
from sklearn.model_selection import GridSearchCV
param_dict = {"n_neighbors":[1,3,5,7,9]}
estimator=GridSearchCV(estimator,param_grid=param_dict,cv=10)
estimator.fit(x_train,y_train)


# 5）评估
# 方法1 ：直接比对真实值和预测值
y_predict = estimator.predict(x_test)
print(y_predict)
print("直接对比真实值和预测值:\n",y_test == y_predict)

# 方法2：计算准确率
score = estimator.score(x_test,y_test)
print("准确率为：\n",score)


# 查看最佳参数
print("最佳参数\n",estimator.best_params_)
# 查看最佳结果
print("最佳结果\n",estimator.best_score_)
# 查看最佳估计器
print("最佳估计器\n",estimator.best_estimator_)



# https://zhuanlan.zhihu.com/p/255222709        KNN来选股票


