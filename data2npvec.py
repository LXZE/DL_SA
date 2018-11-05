#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob, re, time, sys
import argparse
import itertools
import pandas as pd
import numpy as np
from subprocess import call
import multiprocessing as mp
import pythainlp as pyt
from gensim.models import KeyedVectors
import clean
sys.path.append('./utility')
from ThaiTextUtility import ThaiTextUtility as ttext

model_path = 'thai-word-segmentation/saved_model'

getTime = lambda : time.strftime("%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser(description='process text to npy')
parser.add_argument('-n', '--no_tensor', nargs='?', const=True, help='use when no tensorflow installed')
args, leftovers = parser.parse_known_args()

pos_name = 'pos'
neg_name = 'wn_neg'

# import data with label
# fileList = ['data/test.txt', '../dataset/pos.txt', '../dataset/neg.txt']
fileList = [f'../dataset/{pos_name}.txt', f'../dataset/{neg_name}.txt']
files = []
for idx, fileName in enumerate(fileList):
	open_file = open(fileName, 'r')
	tmp = []
	for line in open_file.readlines():
		tmp.append([1 if idx == 0 else 0, line[:-1]])
	files.append(pd.DataFrame(tmp, columns=['label', 'text']))

# word tokenize
pos_df = files[0]
neg_df = files[1]

def nonzero(a):
	return [i for i,e in enumerate(a) if e != 0]
def split(s, indices):
	return [s[i:j] for i,j in zip(indices, indices[1:]+[None])]

if args.no_tensor is None:
	import tensorflow as tf
	sys.path.append('./thai-word-segmentation')
	from thainlplib import ThaiWordSegmentLabeller as tlabel

	sess = tf.Session()
	model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)

	graph = tf.get_default_graph()
	g_inputs = graph.get_tensor_by_name('IteratorGetNext:1')
	g_lengths = graph.get_tensor_by_name('IteratorGetNext:0')
	g_training = graph.get_tensor_by_name('Placeholder_1:0')
	g_outputs = graph.get_tensor_by_name('boolean_mask_1/Gather:0')

util = ttext()

def tokenize(df):
	tokenized_sentence = []

	# for idx, row in itertools.islice(df.iterrows(), 2):
	for idx, row in df.iterrows():
		test_input = row['text']
		test_input = clean.fixing(test_input)

		if args.no_tensor is None:
			inputs = [tlabel.get_input_labels(test_input)]
			len_input = [len(test_input)]

			# TODO: check performance between newmm, bi-lstm and deepcut and
			result = sess.run(g_outputs,
				feed_dict={g_inputs: inputs, g_lengths: len_input, g_training: False})
			cut_word = split(test_input, nonzero(result))

		else:
			cut_word = pyt.word_tokenize(test_input, engine='newmm')

		cut_word = list(map(lambda x: clean.clean_word(x), cut_word))
		# TODO: make suggestion word substitution depends on variable
		suggest_word = list(map(lambda x: util.lemmatize(x), cut_word))
		for idx, (word, alt) in enumerate(zip(cut_word, suggest_word)):
			if(len(alt) == 1):
				cut_word[idx] = alt[0][0]
		tokenized_sentence.append(cut_word)

	return tokenized_sentence

ncore = mp.cpu_count()
pool = mp.Pool(ncore)
# df_split = np.array_split(pos_df, ncore, axis=1)
# NOTE: senquence in list of sentence is not in order after distributed through pool
dfs = [pos_df[i::ncore] for i in range(ncore)]
list_sentence_pos = sum(pool.map(tokenize, dfs), [])
dfs = [neg_df[i::ncore] for i in range(ncore)]
list_sentence_neg = sum(pool.map(tokenize, dfs), [])
pool.close()
pool.join()

# then let input length = 100
# vocab size maybe 60000+2+n (n can be found from traverse through our data set)
# weight of pre-trained database will be imported and added with new vocab
# new vocab's vector can generate with all zeros, all random, average from top nearest k words
# if use keras then make it trainable

# word vectorize
# TODO: load this model and train it again (transfer learning)
# put unfounded vocab by averaging a value or randomized
vector_model_dir = '../model/thai2vec/word2vec/' 
vector_model_path = f'{vector_model_dir}thai2vec02.bin'

vector_model = KeyedVectors.load_word2vec_format(vector_model_path, binary=True)

word_dict = {}
for word in vector_model.index2word:
	word_dict[word] = vector_model[word]
word_vec = pd.DataFrame.from_dict(word_dict, orient='index')
words = vector_model.index2word

input_len = 100
pad_token = '_pad_'
unk_token = '_unk_'
spc_token = '_space_'

def sub_space(tok_sentence):
	return list(map(lambda token: '_space_' if token == ' ' else token, tok_sentence))

def pad_sentence(tok_sentence):
	return tok_sentence + [pad_token] * (input_len - len(tok_sentence))

def sub_unk(tok_sentence):
	return list(map(lambda token: unk_token if token not in words else token, tok_sentence))

def rm_unk(tok_sentence):
	return list(filter(lambda token: token in words, tok_sentence))

def rm_dup_spc(tok_sentence):
	res = []
	for token in tok_sentence:
		if token == spc_token and res[-1] == token:
			print(res)
			pass
		else:
			res.append(token)
	return res

# 2 approaches
# 1: sub all unknown word with token _unk_ before pad
# 2: ignore all unknown word then padding to input len
approach = 1
def token_prepare(tok_sentence):
	tmp = tok_sentence
	if approach == 1:
		tmp = sub_space(tmp)
		tmp = pad_sentence(tmp)
		tmp = sub_unk(tmp)
	else:
		tmp = rm_unk(tmp)
		tmp = rm_dup_spc(tmp)
		tmp = pad_sentence(tmp)
	return tmp
list_sentence_pos = list(map(lambda sen: token_prepare(sen), list_sentence_pos))
list_sentence_neg = list(map(lambda sen: token_prepare(sen), list_sentence_neg))

def sen2vec(tok_sentence):
	tmp = []
	for token in tok_sentence:
		tmp.append(word_dict[token])
	return tmp

list_vector_pos = np.array(list(map(lambda sen: sen2vec(sen), list_sentence_pos)))
list_vector_neg = np.array(list(map(lambda sen: sen2vec(sen), list_sentence_neg)))

np.save(f'{pos_name}.npy', list_vector_pos)
np.save(f'{neg_name}.npy', list_vector_neg)

# TODO: make data structure suitable for lstm and can be transfer to anywhere as numpy format
# NOTE: Compare between (fix oov|mean oov|ignore oov)*(newmm|bi-lstm|deepcut)*(thai2vec|our embed)*(twit data|wongnai data)
# save vector as numpy for colab
