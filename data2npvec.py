#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob, re, time
import pandas as pd
import numpy as np
import tensorflow as tf
from subprocess import call
import pythainlp as pyt
from clean_parallel import fixing
import sys
sys.path.append('./ThaiTextUtility')
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
repeat_pattern = re.compile(r'([^\d\s]+?)\1+')
repeatable_char = ['อ', 'ร', 'ว', 'ย']
line_sub = lambda line, match, count: re.sub('{}'.format(match.group(0)), match.group(1)*count, line, count=1)
def clean_word(word):
    line = word
    try:
	    if bool(repeat_pattern.search(line)):
		    for match in repeat_pattern.finditer(line):
			    if match.group(1) in repeatable_char: # if match char set in repeatable list
				    pos = re.search(match.group(0),line).start()
				    if line[pos-1] in ['เ', 'แ'] or match.group(1) == 'ร': # if char before that repeat char is e|er-vowel or it's 'r' then twice
					    line = line_sub(line, match, 2)
				    else: # if char before not match condition then it should be once
					    line = line_sub(line, match, 1)
			    else: # if match char set not in repeatable char
				    if len(match.group(1)) == 1: # if not repeat char but match len = 1 so repeat once
					    line = line_sub(line, match, 1)
				    else: # if repeat char is kind of pattern then twice
					    line = line_sub(line, match, 2)
			    line = line_sub(line, match, 2)
	    else:
		    pass
    except AttributeError:
	    pass
    return line

cleaner = ttext.ThaiTextUtility()
for idx, row in pos_df.iterrows():
    test_input = row['text']
    inputs = [tlabel.get_input_labels(test_input)]
    len_input = [len(test_input)]
    # print(inputs, len_input)

    result = sess.run(g_outputs,
        feed_dict={g_inputs: inputs, g_lengths: len_input, g_training: False})

    cut_word = split(test_input, nonzero(result))
    cut_word = list(filter(lambda x: len(set(x)) > 1 or x == ' ', cut_word))
    cut_word = list(map(lambda x: clean_word(x), cut_word))
    suggest_word = list(map(lambda x: cleaner.lemmatize(x), cut_word))
    
    for idx, (word, alt) in enumerate(zip(cut_word, suggest_word)):
        # print(word, alt)
        if(len(alt) == 1):
            cut_word[idx] = alt[0][0]
    print('Before: ', row['text'])
    print('After: ',''.join(cut_word))
    print()

# word vectorize

# save vector as numpy for colab
