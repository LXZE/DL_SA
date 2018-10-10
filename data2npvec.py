#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob, re, time, sys
import pandas as pd
import numpy as np
import tensorflow as tf
from subprocess import call
import pythainlp as pyt
from gensim.models import KeyedVectors
import clean
sys.path.append('./utility')
from ThaiTextUtility import ThaiTextUtility as ttext
sys.path.append('./thai-word-segmentation')
from thainlplib import ThaiWordSegmentLabeller as tlabel

model_path = 'thai-word-segmentation/saved_model'

getTime = lambda : time.strftime("%Y-%m-%d %H:%M:%S")

# import data with label
# fileList = sorted(glob.glob('../dataset/'))
# fileList = ['data/test.txt', '../dataset/pos.txt', '../dataset/neg.txt']
fileList = ['../dataset/pos.txt', '../dataset/neg.txt']
files = []
for idx, fileName in enumerate(fileList):
	open_file = open(fileName, 'r')
	tmp = []
	for line in open_file.readlines():
		tmp.append([1 if idx == 0 else 0, line[:-1]])
	files.append(pd.DataFrame(tmp, columns=['label', 'text']))
# print(files)

# word tokenize
pos_df = files[0]
neg_df = files[1]
sess = tf.Session()
model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)

graph = tf.get_default_graph()
g_inputs = graph.get_tensor_by_name('IteratorGetNext:1')
g_lengths = graph.get_tensor_by_name('IteratorGetNext:0')
g_training = graph.get_tensor_by_name('Placeholder_1:0')
g_outputs = graph.get_tensor_by_name('boolean_mask_1/Gather:0')

def nonzero(a):
	return [i for i,e in enumerate(a) if e != 0]
def split(s, indices):
	return [s[i:j] for i,j in zip(indices, indices[1:]+[None])]

util = ttext.ThaiTextUtility()
for idx, row in pos_df.iterrows():
	test_input = row['text']
	# TODO: add more duplicate character cleaning here
	test_input = clean.fixing(test_input)
	inputs = [tlabel.get_input_labels(test_input)]
	len_input = [len(test_input)]

	result = sess.run(g_outputs,
		feed_dict={g_inputs: inputs, g_lengths: len_input, g_training: False})

	cut_word = split(test_input, nonzero(result))
	cut_word = list(map(lambda x: clean.clean_word(x), cut_word))
	suggest_word = list(map(lambda x: util.lemmatize(x), cut_word))

	for idx, (word, alt) in enumerate(zip(cut_word, suggest_word)):
		if(len(alt) == 1):
			cut_word[idx] = alt[0][0]
	print('Before: ', row['text'])
	print(cut_word)
	print('After: ',''.join(cut_word))
	print()

# word vectorize
# vector_model_dir = '../model/thai2vec/word2vec/' 
# vector_model_path = f'{vector_model_dir}thai2vec02.bin'

# vector_model = KeyedVectors.load_word2vec_format(vector_model_path, binary=True)

# word_dict = {}
# for word in vector_model.index2word:
# 	word_dict[word] = vector_model[word]
# word_vec = pd.DataFrame.from_dict(word_dict, orient='index')

# save vector as numpy for colab
