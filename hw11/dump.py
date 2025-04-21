from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
# print(iris)
print('iris.keys()=', bc.keys())
print('filename=', bc.filename) # 特徵屬性的名稱
print('feature_names=', bc.feature_names) # 特徵屬性的名稱
print('data=', bc.data) # data 是一個 numpy 2d array, 通常用 X 代表
print('target=', bc.target) # target 目標值，答案，通常用 y 代表
print('target_names=', bc.target_names) # 目標屬性的名稱
print('DESCR=', bc.DESCR)

import pandas as pd

x = pd.DataFrame(bc.data, columns=bc.feature_names)
print('x=', x)
y = pd.DataFrame(bc.target, columns=['target'])
print('y=', y)
data = pd.concat([x,y], axis=1) # 水平合併 x | y
# axis 若為 0 代表垂直合併  x/y (x over y)
print('data=\n', data)
data.to_csv('bc_dump.csv', index=False)
