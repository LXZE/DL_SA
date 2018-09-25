import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from subprocess import call
import time
import pythainlp as pyt
from clean_parallel import fixing
import sys
sys.path.append('./ThaiTextUtility')
sys.path.append('./thai-word-segmentation')
from ThaiTextUtility import ThaiTextUtility as ttext
from thainlplib import ThaiWordSegmentLabeller as tlabel

model_path = 'thai-word-segmentation/saved_model'

getTime = lambda : time.strftime("%Y-%m-%d %H:%M:%S")

# import data with label
# fileList = sorted(glob.glob('../dataset/'))
fileList = ['data/test.txt', '../dataset/pos.txt', '../dataset/neg.txt']
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

for idx, row in pos_df.iterrows():
    test_input = row['text']
    inputs = [tlabel.get_input_labels(test_input)]
    len_input = [len(test_input)]
    print(inputs, len_input)

    result = sess.run(g_outputs,
        feed_dict={g_inputs: inputs, g_lengths: len_input, g_training: False})

    for word in split(test_input, nonzero(result)): print(word, end='|')
    print()

for idx, row in pos_df.iterrows():
	# res = pyt.word_tokenize(row['text'], engine='newmm')
	# print(cleaner.lemmatize(row['text']))
	# print('-'*30)
	pass
	# pyt.word_tokenize(line, engine='newmm')

# word vectorize

# save vector as numpy for colab
