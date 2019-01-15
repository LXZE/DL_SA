import glob, re, time, sys, os
import pandas as pd
import numpy as np
import dill as pickle
from subprocess import call
from gensim.models import KeyedVectors
from collections import defaultdict as ddict
from itertools import islice
from dotenv import load_dotenv
import clean

load_dotenv()

# first load and save
# vector_model_dir = '../model/thai2vec/word2vec/'
# vector_model_path_bin = f'{vector_model_dir}thai2vec02.bin'
# vector_model_path = f'{vector_model_dir}thai2vec02'

# vector_model = KeyedVectors.load_word2vec_format(vector_model_path_bin, binary=True)

# word_dict = {}
# word_dict['_lol_'] = vector_model['555']
# for word in vector_model.index2word:
# 	word_dict[word] = vector_model[word]
# word_vec = pd.DataFrame.from_dict(word_dict, orient='index')

# new_dir = '../model/dl_sa/'
# word_vec.to_csv(f'{new_dir}vec.vec', sep=' ', header=False, line_terminator='\n', encoding='utf-8')
# print('saved')
# shape_val = re.search('(\d+)\,\s(\d+)', str(word_vec.shape)).groups()
# shape_str = f'{shape_val[0]} {shape_val[1]}'

# with open(f'{new_dir}vec.vec', 'r+') as f:
# 	content = f.read()
# 	f.seek(0, 0)
# 	f.write(shape_str.rstrip('\r\n') + '\n' + content)

# model = KeyedVectors.load_word2vec_format(f'{new_dir}vec.vec', binary=False, unicode_errors='ignore')
# model.save_word2vec_format(f'{new_dir}vec.bin', None, True)

# itos = model.index2word
# stoi = ddict(lambda: 0, {v:k for k,v in enumerate(itos)})
# print(len(itos))

# current load
vector_model_dir = '../model/dl_sa/'
vector_model_path_bin = f'{vector_model_dir}vec.bin'
vector_model = KeyedVectors.load_word2vec_format(vector_model_path_bin, binary=True)
itos = vector_model.index2word
stoi = ddict(lambda: 0, {v:k for k,v in enumerate(itos)})

# embedding_dim = 300
# embedding_matrix = np.zeros((len(itos), embedding_dim))
# for key, vec in word_dict.items():
# 	embedding_matrix[stoi[key]] = vec
# np.save('../model/vec.npy', embedding_matrix)

pos_name = os.getenv('pos_file_name')
neg_name = os.getenv('neg_file_name')

pos_tok = pickle.load(open(f'../dataset/{pos_name}_tok.pkl', 'rb'))
neg_tok = pickle.load(open(f'../dataset/{neg_name}_tok.pkl', 'rb'))

input_len = 600
pad_token = '_pad_'
unk_token = '_unk_'

def sub_space(tok_sentence):
	return list(map(lambda token: '_space_' if token == ' ' else token, tok_sentence))
def sub_lol(tok_sentence):
	return list(map(lambda token: '_lol_' if token == 'lol' else token, tok_sentence))
def pad_sentence(tok_sentence):
	if len(tok_sentence) > input_len:
		return tok_sentence[:input_len]
	else:
		return tok_sentence + [pad_token] * (input_len - len(tok_sentence))

unknown_words = {}
def sen2int(tok_list):
	global unknown_words
	tok_sentence = np.array(tok_list)
	new_int_sentence = []
	for list_tok in tok_sentence:
		tmp = []
		list_tok = sub_space(list_tok)
		list_tok = sub_lol(list_tok)
		list_tok = pad_sentence(list_tok)
		for tok in list_tok:
			if tok in itos:
				tmp.append(stoi[tok])
			else:
				if tok not in unknown_words:
					unknown_words[tok] = 1
				else:
					unknown_words[tok] += 1
				tmp.append(stoi[unk_token])
		new_int_sentence.append(tmp)
	return new_int_sentence

pos_int = np.array(sen2int(pos_tok))
neg_int = np.array(sen2int(neg_tok))

# print(pos_int.shape, neg_int.shape)
np.save(f'../dataset/{pos_name}_int.npy', pos_int)
np.save(f'../dataset/{neg_name}_int.npy', neg_int)
# print(unknown_words, len(unknown_words))
pickle.dump(unknown_words, open('../dataset/unknown.pkl', 'wb'))