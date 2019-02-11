import os
import numpy as np
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

pos_name = os.getenv('pos_file_name')
neg_name = os.getenv('neg_file_name')
prefix = ''

pos = np.load(f'../dataset/{prefix}{pos_name}_int.npy')
neg = np.load(f'../dataset/{prefix}{neg_name}_int.npy')

print(f'pos size = {len(pos)}')
print(f'neg size = {len(neg)}')

x = np.concatenate((pos,neg), axis = 0)
y = np.concatenate(
	(
		np.full((pos.shape[0], 2), [0, 1]),
		np.full((neg.shape[0], 2), [1, 0])
	), axis = 0
)
x = x.astype('int32')
y = y.astype('int32')

x_train, x_test, y_train, y_test =  train_test_split(x,y, test_size=0.15, stratify=y, random_state=420)

print(x_train, y_train)
print(x_train.shape, y_train.shape)
print(x_train.dtype, y_train.dtype)
print(x_test, y_test)
print(x_test.shape, y_test.shape)
print(x_test.dtype, y_test.dtype)

print(f'pos size = { len(y_test[np.where( y_test[:,0] == 0 )]) }')
print(f'neg size = { len(y_test[np.where( y_test[:,0] == 1 )]) }')

all_data = [x_train, x_test, y_train, y_test]
np.save('../dataset/train_test_data.npy', all_data)
print('-'*30)
# validation
[vald_x_train, vald_x_test, vald_y_train, vald_y_test] = np.load('../dataset/train_test_data.npy')

print(vald_x_train, vald_y_train)
print(vald_x_train.shape, vald_y_train.shape)
print(vald_x_train.dtype, vald_y_train.dtype)
print(vald_x_test, vald_y_test)
print(vald_x_test.shape, vald_y_test.shape)
print(vald_x_test.dtype, vald_y_test.dtype)

np.testing.assert_array_equal(vald_x_train, x_train)
np.testing.assert_array_equal(vald_x_test, x_test)
np.testing.assert_array_equal(vald_y_train, y_train)
np.testing.assert_array_equal(vald_y_test, y_test)

print('pass test')

# kfold
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=10, random_state=42, shuffle=True)
# for train_idx, val_idx in kf.split(x_train):
# 	cv_x_train = x_train[train_idx]
# 	cv_y_train = y_train[train_idx]

# 	cv_x_val = x_train[val_idx]
# 	cv_y_val = y_train[val_idx]

# 	print(cv_x_train[:5], cv_y_train[:5])
# 	print(cv_x_val[:5], cv_y_val[:5])
# 	print("TRAIN:", train_idx[:5], "TEST:", val_idx[:5])
# 	print("TRAIN len:", len(train_idx), "TEST len:", len(val_idx))