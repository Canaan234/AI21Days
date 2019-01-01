import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def ValidateSingleIsRight( _validate_value, _real_value, _ErrorTolerance=0.05):
    if abs(_real_value-_validate_value)/_real_value<=_ErrorTolerance:
        return True
    else:
        return False
    
def ValidateRightRate(_validate_set, _real_set, _set_name):
    right_num=0
    for index in range(len(_real_set)):
        if ValidateSingleIsRight(_validate_value=_validate_set[index], _real_value=_real_set[index]):
            right_num+=1
    print("%s rightNum:%d" %(_set_name, right_num))
    #预测准确率=（预测正确样本数）/（总测试样本数）* 100%
    right_rate = right_num/len(_real_set)
    print("%s rightRate:%f" %(_set_name,right_rate))
    return right_rate


data = pd.read_csv("msft_stockprices_dataset.csv") 
used_features = [ "High Price", "Low Price", "Open Price", "Volume"]
X=data[used_features].values
close_price = data["Close Price"].values
#ID作为横坐标
plt_x = data["Date"].values
#print(plt_x)
rows_count = len(data)-1 #去掉标题行共1008行
rows_count_training = rows_count//10*7
rows_count_validate = rows_count_training+rows_count//10*1
#按7:1:2或2:1:1分训练、验证、测试集
X_TrainingSet = X[:rows_count_training]
X_ValidateSet = X[rows_count_training:rows_count_validate]
X_TestSet = X[rows_count_validate:] 

plt_x_validate_set = plt_x[rows_count_training:rows_count_validate]

# 把目标数据（特征对应的真实值）也分为训练集、验证集和测试集
Y_TrainingSet = close_price[:rows_count_training]
Y_ValidateSet = close_price[rows_count_training:rows_count_validate]
Y_TestSet = close_price[rows_count_validate:]

plt_x_test_set = plt_x[rows_count_validate:]
# 创建线性回归模型
regr = linear_model.LinearRegression()

# 用训练集训练模型——看就这么简单，一行搞定训练过程
regr.fit(X_TrainingSet, Y_TrainingSet)

# 用训练得出的模型进行预测
diabetes_y_pred = regr.predict(X_ValidateSet)

#输出验证集中数据，作为参考，字典中的key值即为csv中列名
dataframe = pd.DataFrame({'ValidateSet':Y_ValidateSet,'diabetes_y_pred':diabetes_y_pred})
#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("ValidateSet.csv",index=False,sep=',')

#人工指定一个 ErrorTolerance（一般是10%或者5%），当 |预测值-真实值| / 真实值 <= ErrorTolerance 时，我们认为预测正确，否则为预测错误。
ErrorTolerance = 0.05
validate_right_rate = ValidateRightRate(_validate_set=diabetes_y_pred, _real_set=Y_ValidateSet, _set_name='Validate')
#用测试集继续测试
diabetes_y_pred_test = regr.predict(X_TestSet)
test_right_rate = ValidateRightRate(_validate_set=diabetes_y_pred_test, _real_set=Y_TestSet, _set_name='Test')
#输出验证集中数据，作为参考，字典中的key值即为csv中列名
dataframe = pd.DataFrame({'TestSet':Y_TestSet,'diabetes_y_pred':diabetes_y_pred_test})
#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("TestSet.csv",index=False,sep=',')


# 将测试结果以图标的方式显示出来
plt.scatter(plt_x_test_set, Y_TestSet,  color='black')
plt.plot(plt_x_test_set, diabetes_y_pred_test, color='blue', linewidth=2)

plt.xticks(())
plt.yticks(())

plt.show()
#print(data)


