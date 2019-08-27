import glob
import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from keras.models import load_model
from keras import backend as K
from attention import AttentionWithContext as att
from sklearn import metrics

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

# get model file list
model_path = '../model/result/*'
model_dir_list = sorted(glob.glob(model_path))
model_file_list = []
for dirname in model_dir_list:
	if os.name == 'nt':
		model_type = dirname.split('\\')[-1]
	else:
		model_type = dirname.split('/')[-1]
	model_file_list.append({
		'name': model_type,
		'path': glob.glob(dirname + '/*.hdf5')[0]
	})
# print(model_file_list)

# load data
data_path = '../dataset/train_test_data.npy'
[x_train, x_test, y_train, y_test] = np.load(data_path)
get_val = lambda arr: list(map(lambda x: x[1], arr))
y_true = get_val(y_test)

# split data to short or long
pad_i = 4999
get_first_pad_idx = lambda elem: np.where(elem == pad_i)[0]
new_x_test_nopad = list(map(lambda elem: np.delete(elem, get_first_pad_idx(elem)), x_test))
new_x_test_len = list(map(lambda x: len(x), new_x_test_nopad))

mean = 120
short_x_test, short_y_test = [], []
long_x_test, long_y_test = [], []

for x,y in zip(x_test, y_test):
	first_pad_idx = get_first_pad_idx(x)
	try:
		first_pad_idx = first_pad_idx[0]
	except:
		first_pad_idx = len(x_test[0])
	
	if first_pad_idx < mean:
		short_x_test.append(x)
		short_y_test.append(y)
	else:
		long_x_test.append(x)
		long_y_test.append(y)
short_y_true = get_val(short_y_test)
long_y_true = get_val(long_y_test)
short_x_test = np.array(short_x_test)
long_x_test = np.array(long_x_test)
short_y_test = np.array(short_y_test)
long_y_test = np.array(long_y_test)

# print(short_x_test.shape, long_x_test.shape)

# convert data to text
# from collections import defaultdict as ddict
# vec_path = '../model/dl_sa/vec.bin'
# vector_model = KeyedVectors.load_word2vec_format(vec_path, binary=True)
# itos = vector_model.index2word
# def sub_space(tok_sentence):
# 	return list(map(lambda token: ' ' if token == '_space_' else token, tok_sentence))
# def sub_lol(tok_sentence):
# 	return list(map(lambda token: '555' if token == '_lol_' else token, tok_sentence))
# def sub_unknown(tok_sentence):
# 	return list(map(lambda token: '?' if token == '_unk_' else token, tok_sentence))
# def rm_pad(tok_sentence):
# 	return list(map(lambda token: '' if token == '_pad_' else token, tok_sentence))
# x_test_sentence = []
# x_test = x_test.astype('int')
# for arr_tok in x_test:
# 	tmp = list(filter(lambda val: val != 2, arr_tok))
# 	tmp = list(map(lambda val: itos[val], tmp))
# 	tmp = sub_space(tmp)
# 	tmp = sub_lol(tmp)
# 	tmp = sub_unknown(tmp)
# 	tmp = rm_pad(tmp)
# 	x_test_sentence.append(''.join(tmp))
# print(x_test_sentence[:10])

df_short = pd.DataFrame({}, columns=['model', 'precision', 'recall', 'f1'])
df_long = pd.DataFrame({}, columns=['model', 'precision', 'recall', 'f1'])
# print(df)
patt = lambda a,b,c,d: {'model':a, 'precision':b, 'recall':c, 'f1':d}

res = {'pos':1, 'neg':0}
for model_data in model_file_list:

	print(f'{model_data["name"]} start')

	if model_data['name'].find('attn') > 0:
		model = load_model(str(model_data['path']), custom_objects={'AttentionWithContext': att})
	else:
		model = load_model(str(model_data['path']))
	
	yy = model.predict_classes(x_test)
	np.save(f'predicted/{model_data["name"]}.npy', yy)

'''
	short_y_predict = model.predict_classes(short_x_test)
	long_y_predict = model.predict_classes(long_x_test)
	
	short_score = model.evaluate(short_x_test, short_y_test, batch_size=50)
	long_score = model.evaluate(long_x_test, long_y_test, batch_size=50)
	# df[model_data['name']] = y_predict

	df_short = df_short.append(
		patt(model_data['name'],
			metrics.precision_score(short_y_true, short_y_predict),
			metrics.recall_score(short_y_true, short_y_predict),
			metrics.f1_score(short_y_true, short_y_predict)
		), ignore_index=True
	)

	df_long = df_long.append(
		patt(model_data['name'],
			metrics.precision_score(long_y_true, long_y_predict),
			metrics.recall_score(long_y_true, long_y_predict),
			metrics.f1_score(long_y_true, long_y_predict)
		), ignore_index=True
	)

	with open(f'../model/{model_data["name"]}_test_result.txt', 'w') as f:
		f.write('{}\n'.format('---short---'))
		f.write('{}: {}\n'.format(model.metrics_names[0], short_score[0]))
		f.write('{}: {}%\n'.format(model.metrics_names[1], short_score[1]*100))
		f.write('{}: {}\n'.format('Report', metrics.classification_report(short_y_true, short_y_predict, target_names = ['negative', 'positive'], digits=6)))
		
		f.write('{}\n'.format('---long---'))
		f.write('{}: {}\n'.format(model.metrics_names[0], long_score[0]))
		f.write('{}: {}%\n'.format(model.metrics_names[1], long_score[1]*100))
		f.write('{}: {}\n'.format('Report', metrics.classification_report(long_y_true, long_y_predict, target_names = ['negative', 'positive'], digits=6)))
'''

	K.clear_session()
	print(f'{model_data["name"]} finish')
# print(df)
# df.to_csv('./result_eval.csv')
# print(df_short)
df_short.to_csv('../model/short.csv')
df_long.to_csv('../model/long.csv')