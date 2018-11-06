import multiprocessing as mp
import numpy as np
import pandas as pd
import sys, re
import clean

def isEnglishOnly(s):
	try:
		s.encode(encoding='utf-8').decode('ascii')
	except UnicodeDecodeError:
		return False
	else:
		return True

def wash(df):
	list_sentence = []
	for idx, val in df.iteritems():
		tmp = clean.cleanLine(val)
		tmp = clean.filtering(tmp)
		tmp = clean.fixing(tmp)
		tmp = clean.stripping(tmp)

		if isEnglishOnly(val):
			continue

		list_sentence.append(tmp)
	return list_sentence

dataset_path = '../dataset/wongnai/w_review_train.csv'
df = pd.read_csv(dataset_path, delimiter=';', header=None,
	names=['text','label'], index_col=False)

df_seperated = []
for i in range(1,6):
	tmp = pd.DataFrame(df[df['label'] == i]).reset_index()
	df_seperated.append(tmp['text'])
for idx, df_elem in enumerate(df_seperated):
	df_seperated[idx] = df_elem.apply(lambda x: x.replace('\n', ' '))

neg_df = pd.concat(df_seperated[0:2])
pos_df = df_seperated[4]

data_pos = wash(pos_df)
data_neg = wash(neg_df)

file = open('../dataset/wn_pos.txt', 'w')
file.writelines('\n'.join(data_pos))
file.close()

file = open('../dataset/wn_neg.txt', 'w')
file.writelines('\n'.join(data_neg))
file.close()