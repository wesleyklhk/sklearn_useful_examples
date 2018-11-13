from sklearn.model_selection import PredefinedSplit
import pandas as pd
import numpy as np

########dataset
X1 = np.arange(1,11)
X2 = X1**2

y = X1 - 1

X = pd.DataFrame({'X1':X1,'X2':X2}).values
print('dataset')
print('X')
print(X)

print('y')
print(y)
print()
print()
print('------------------------------')

########manually splitting into training and validation set
test_fold = [-1] * 6 + [0] * 4
#-1 indicates that index will belong to the training set
#in this case, the first 6 records will belong to the training set
#others will belong to validation set
ps = PredefinedSplit(test_fold)


#########splitted
print('splitted training and validation set')
for train_index,test_index in ps.split():
	print('train_index',train_index)
	print('X_train (first 6 records of X)')
	print(X[train_index])
	print('y_train (first 6 records of y)')
	print(y[train_index])

	print()
	print()

	print('test_index',test_index)
	print('X_test (last 4 records of X)')
	print(X[test_index])
	print('y_test (last 4 records of y)')
	print(y[test_index])	


