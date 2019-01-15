import numpy as np
from sklearn.model_selection import train_test_split

pos = np.load('../dataset/wn_pos_int.npy')
neg = np.load('../dataset/wn_neg_int.npy')


x = np.concatenate((pos,neg), axis = 0)
y = np.concatenate(
	(
		np.full((pos.shape[0], 2), [0,1]),
		np.full((neg.shape[0], 2), [1,0])
	), axis = 0
)

# length = []
# print(x.shape, y.shape)

# for idx, val in enumerate(x):
# 	length.append(len(val))
# length = np.array(length)
# print(length.mean(), length.std())
x_train, x_test, y_train, y_test =  train_test_split(x,y, test_size=0.2)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=42, shuffle=True)
for train_idx, val_idx in kf.split(x_train):
	cv_x_train = x_train[train_idx]
	cv_y_train = y_train[train_idx]

	cv_x_val = x_train[val_idx]
	cv_y_val = y_train[val_idx]

	print(cv_x_train[:5], cv_y_train[:5])
	print(cv_x_val[:5], cv_y_val[:5])
	print("TRAIN:", train_idx[:5], "TEST:", val_idx[:5])
	print("TRAIN len:", len(train_idx), "TEST len:", len(val_idx))