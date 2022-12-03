# 转换器：特征工程的一个父类
# 1.实例化一个转换器类
# 2.fit_transform             # 计算

# 估计器
# 所有的机器学习算法都被封装到其中
# 1.实例化一个estimator
# 2.estimator.fit(x_train,y_train)           # 计算训练，调用完毕就意味着模型生成
# 3.模型评估 ：
# 1）直接比对真实值和预测值
# y_predict = estimator.predict(x_test)         # 求出预测值
# y_test == y_predict                           # 比对真实值和预测值
# 2） 计算准确率
# estimator.score(x_test,y_test)               # 把测试集的特征值和目标值都传进来




