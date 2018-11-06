import numpy as np
from sklearn.model_selection import train_test_split

pos = np.load('../model/pos.npy')
neg = np.load('../model/neg.npy')

x = np.concatenate((pos,neg), axis = 0)
y = np.concatenate(
	(
		np.full((pos.shape[0], 2), [0,1]),
		np.full((neg.shape[0], 2), [1,0])
	), axis = 0
)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test =  train_test_split(x,y, test_size=0.2)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)