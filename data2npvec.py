import glob
import pandas as pd
from subprocess import call
import time
import pythainlp as pyt
from clean_parallel import fixing
import sys
sys.path.append('./ThaiTextUtility')
from ThaiTextUtility import ThaiTextUtility as ttext

getTime = lambda : time.strftime("%Y-%m-%d %H:%M:%S")

# import data with label
# fileList = sorted(glob.glob('../dataset/'))
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
pos_file = files[0]
neg_file = files[1]
print(dir(ttext))

for idx, row in pos_file.iterrows():
	# res = pyt.word_tokenize(row['text'], engine='newmm')
	# print(cleaner.lemmatize(row['text']))
	# print('-'*30)
	pass
	# pyt.word_tokenize(line, engine='newmm')

# word vectorize

# save vector as numpy for colab