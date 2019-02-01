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

vector_model_dir = '../model/dl_sa/'
vector_model_path_bin = f'{vector_model_dir}vec.bin'
vector_model = KeyedVectors.load_word2vec_format(vector_model_path_bin, binary=True)
itos = vector_model.index2word
stoi = ddict(lambda: 0, {v:k for k,v in enumerate(itos)})

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
		list_tok = list(map(lambda word: clean.stripping(word), list_tok))
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