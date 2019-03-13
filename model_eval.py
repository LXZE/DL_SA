import glob
import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from keras.models import load_model
from keras import backend as K
from attention import AttentionWithContext as att

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

# convert data to text
from collections import defaultdict as ddict
vec_path = '../model/dl_sa/vec.bin'
vector_model = KeyedVectors.load_word2vec_format(vec_path, binary=True)
itos = vector_model.index2word
def sub_space(tok_sentence):
	return list(map(lambda token: ' ' if token == '_space_' else token, tok_sentence))
def sub_lol(tok_sentence):
	return list(map(lambda token: '555' if token == '_lol_' else token, tok_sentence))
def sub_unknown(tok_sentence):
	return list(map(lambda token: '?' if token == '_unk_' else token, tok_sentence))
def rm_pad(tok_sentence):
	return list(map(lambda token: '' if token == '_pad_' else token, tok_sentence))
x_test_sentence = []
x_test = x_test.astype('int')
for arr_tok in x_test:
	tmp = list(filter(lambda val: val != 2, arr_tok))
	tmp = list(map(lambda val: itos[val], tmp))
	tmp = sub_space(tmp)
	tmp = sub_lol(tmp)
	tmp = sub_unknown(tmp)
	tmp = rm_pad(tmp)
	x_test_sentence.append(''.join(tmp))
# print(x_test_sentence[:10])

df = pd.DataFrame({'text':x_test_sentence, 'target': y_true}, columns=['text', 'target'])
print(df)
res = {'pos':1, 'neg':0}
for model_data in model_file_list:

	if model_data['name'].find('attn') > 0:
		model = load_model(str(model_data['path']), custom_objects={'AttentionWithContext': att})
	else:
		model = load_model(str(model_data['path']))
	
	y_predict = model.predict_classes(x_test)

	df[model_data['name']] = y_predict

	K.clear_session()

print(df)
df.to_csv('./result_eval.csv')