import glob, re, time, sys
import pandas as pd
import numpy as np
import tensorflow as tf
from subprocess import call
import pythainlp as pyt
from gensim.models import KeyedVectors
import clean

vector_model_dir = '../model/thai2vec/word2vec/' 
vector_model_path = f'{vector_model_dir}thai2vec02.bin'

vector_model = KeyedVectors.load_word2vec_format(vector_model_path, binary=True)

word_dict = {}
for word in vector_model.index2word:
	word_dict[word] = vector_model[word]
word_vec = pd.DataFrame.from_dict(word_dict, orient='index')

words = vector_model.index2word

w_rank = {}
for i,word in enumerate(words):
	w_rank[word] = i
WORDS = w_rank

thai_letters = 'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤฤๅลฦฦๅวศษสหฬอฮะัาำิีึืุูเแโใไ็่้๊๋์'

def words(text): return re.findall(r'\w+', text.lower())

def P(word): 
	"Probability of `word`."
	# use inverse of rank as proxy
	# returns 0 if the word isn't in the dictionary
	return - WORDS.get(word, 0)

def correction(word): 
	"Most probable spelling correction for word."
	return max(candidates(word), key=P)

def candidates(word): 
	"Generate possible spelling corrections for word."
	print(known([word]), known(edits1(word)), known(edits2(word)), [word])
	return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
	"The subset of `words` that appear in the dictionary of WORDS."
	return set(w for w in words if w in WORDS)

def edits1(word):
	"All edits that are one edit away from `word`."
	letters    = thai_letters
	splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
	deletes    = [L + R[1:]               for L, R in splits if R]
	transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
	replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
	inserts    = [L + c + R               for L, R in splits for c in letters]
	return set(deletes + transposes + replaces + inserts)

def edits2(word): 
	"All edits that are two edits away from `word`."
	return (e2 for e1 in edits1(word) for e2 in edits1(e1))

print(correction('เอไอเอส'))
# print(correction('นะค่ะ'))