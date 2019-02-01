from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from collections import Counter
import dill as pickle
import re
import clean

unknown_words = pickle.load(open('../dataset/unknown_bilstm.pkl', 'rb'))
bilstm_unk_counter = Counter(unknown_words)
min_threshold = 2
for key, count in list(bilstm_unk_counter.items()):
	if count < min_threshold: del bilstm_unk_counter[key]

unknown_words = pickle.load(open('../dataset/unknown_newmm.pkl', 'rb'))
newmm_unk_counter = Counter(unknown_words)
min_threshold = 2
for key, count in list(newmm_unk_counter.items()):
	if count < min_threshold: del newmm_unk_counter[key]

all_counter = bilstm_unk_counter + newmm_unk_counter
del all_counter['	']
del all_counter[' ']
min_threshold = 2
long_word_threshold = 20
for key, count in list(all_counter.items()):
	try:
		if count < min_threshold: del all_counter[key]
		elif key.find(' ') > -1 or key.find('	') > -1:
			old_key = key
			new_key = clean.stripping(key)
			all_counter[new_key] = count
			del all_counter[old_key]
		if len(key) > long_word_threshold: del all_counter[key]
	except ValueError as e:
		print(e, key)
accept_key = []
# for key, val in all_counter.most_common(5000):
for key, val in all_counter.most_common(4997): # -3 which are special token
	accept_key.append(key)

# create new vec.bin
vector_model_dir = '../model/dl_sa/'
vector_model_path_bin = f'{vector_model_dir}vec.bin'
vector_model = KeyedVectors.load_word2vec_format(vector_model_path_bin, binary=True)

word_dict = {}
for key in accept_key:
	word_dict[key] = np.subtract(np.random.random((300)), 0.5).astype(np.float32)
for word in vector_model.index2word:
	word_dict[word] = vector_model[word]
word_vec = pd.DataFrame.from_dict(word_dict, orient='index')

new_dir = '../model/dl_sa/'
file_name = 'vec_v2'
word_vec.to_csv(f'{new_dir}{file_name}.vec', sep=' ', header=False, line_terminator='\n', encoding='utf-8')
print('saved')
shape_val = re.search('(\d+)\,\s(\d+)', str(word_vec.shape)).groups()
shape_str = f'{shape_val[0]} {shape_val[1]}'

with open(f'{new_dir}{file_name}.vec', 'r+') as f:
	content = f.read()
	f.seek(0, 0)
	f.write(shape_str.rstrip('\r\n') + '\n' + content)

model = KeyedVectors.load_word2vec_format(f'{new_dir}{file_name}.vec', binary=False, unicode_errors='ignore')
model.save_word2vec_format(f'{new_dir}{file_name}.bin', None, True)

itos = model.index2word
stoi = ddict(lambda: 0, {v:k for k,v in enumerate(itos)})
print(len(itos))

embedding_dim = 300
embedding_matrix = np.zeros((len(itos), embedding_dim))
for key, vec in word_dict.items():
	embedding_matrix[stoi[key]] = vec
np.save('../model/vec.npy', embedding_matrix)