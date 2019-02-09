#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob, re, time, sys, os
import argparse
import pandas as pd
import numpy as np
from subprocess import call
import multiprocessing as mp
import concurrent.futures as conc
import pythainlp as pyt
import dill as pickle
from dotenv import load_dotenv
import clean
sys.path.append('./utility')
from ThaiTextUtility import ThaiTextUtility as ttext

load_dotenv()

model_path = 'thai-word-segmentation/saved_model'

getTime = lambda : time.strftime("%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser(description='process text to token and save in pkl')
parser.add_argument('-n', '--no_tensor', nargs='?', const=True, help='use when no tensorflow installed')
args, leftovers = parser.parse_known_args()

pos_name = os.getenv('pos_file_name')
neg_name = os.getenv('neg_file_name')

# import data with label
fileList = [f'../dataset/{pos_name}.txt', f'../dataset/{neg_name}.txt']
files = []
for idx, fileName in enumerate(fileList):
	open_file = open(fileName, 'r', encoding='utf-8')
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

# if args.no_tensor is None:
def predict(df, core):
	print(f'loading tensorflow to core {core}')
	import tensorflow as tf
	sys.path.append('./thai-word-segmentation')
	from thainlplib import ThaiWordSegmentLabeller as tlabel

	config = tf.ConfigProto(device_count = {'GPU': 0})
	sess = tf.Session(config=config)
	model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)

	graph = tf.get_default_graph()
	g_inputs = graph.get_tensor_by_name('IteratorGetNext:1')
	g_lengths = graph.get_tensor_by_name('IteratorGetNext:0')
	g_training = graph.get_tensor_by_name('Placeholder_1:0')
	g_outputs = graph.get_tensor_by_name('boolean_mask_1/Gather:0')

	results = []
	for idx, row in df.iterrows():
		test_input = clean.fixing(row['text'])
		inputs = [tlabel.get_input_labels(test_input)]
		len_input = [len(test_input)]
		result = sess.run(g_outputs,
			feed_dict={g_inputs: inputs, g_lengths: len_input, g_training: False})
		cut_word = split(test_input, nonzero(result))
		cut_word = clean_n_sub(cut_word)
		results.append(cut_word)
	return results

util = ttext()

def clean_n_sub(word):
	cut_word = list(map(lambda x: clean.clean_word(x), word))
	suggest_word = list(map(lambda x: util.lemmatize(x), cut_word))
	for idx, (word, alt) in enumerate(zip(cut_word, suggest_word)):
		if(len(alt) == 1):
			cut_word[idx] = alt[0][0]
	cut_word = list(map(lambda word:
		clean.stripping(word) if len(word) > 1 
		else word, cut_word))
	return cut_word

def tokenize(df, core):
	tokenized_sentence = []
	print('core {} started'.format(core))

	if args.no_tensor is None:
		tokenized_sentence = predict(df, core)
	else:
		for idx, row in df.iterrows():
			test_input = clean.fixing(row['text'])
			cut_word = pyt.word_tokenize(test_input, engine='newmm')
			cut_word = clean_n_sub(cut_word)
			tokenized_sentence.append(cut_word)

	print('core {} finished'.format(core))
	return tokenized_sentence

def parallel_exec(df):
	ncore = mp.cpu_count()
	output_list = []
	with conc.ProcessPoolExecutor(max_workers=ncore) as executor:
		workers = {
			executor.submit(tokenize, df[core_idx::ncore], core_idx):
				core_idx for core_idx in range(ncore)
		}
		for worker in conc.as_completed(workers):
			try:
				data = worker.result()
			except Exception as exc:
				print('Error occurred : {}'.format(exc))
			else:
				output_list.append(data)
	return sum(output_list, [])

if __name__ == '__main__':
	print('run pos df')
	list_sentence_pos = parallel_exec(pos_df)
	print('run neg df')
	list_sentence_neg = parallel_exec(neg_df)

	pickle.dump(list_sentence_pos, open(f'../dataset/{pos_name}_tok.pkl', 'wb'))
	pickle.dump(list_sentence_neg, open(f'../dataset/{neg_name}_tok.pkl', 'wb'))