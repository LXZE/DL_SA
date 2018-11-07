import glob, re, time, sys
import pandas as pd
import numpy as np
import dill as pickle
from gensim.models import KeyedVectors
from collections import defaultdict as ddict
from itertools import islice
import clean

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

vector_model_dir = '../model/thai2vec/word2vec/' 
vector_model_path_bin = f'{vector_model_dir}thai2vec02.bin'
vector_model_path = f'{vector_model_dir}thai2vec02'

vector_model = KeyedVectors.load_word2vec_format(vector_model_path_bin, binary=True)

word_dict = {}
word_dict['_lol_'] = vector_model['555']
for word in vector_model.index2word:
	word_dict[word] = vector_model[word]
word_vec = pd.DataFrame.from_dict(word_dict, orient='index')
itos = vector_model.index2word

itos.insert(0, '_lol_')
stoi = ddict(lambda: 0, {v:k for k,v in enumerate(itos)})

pos_tok = pickle.load(open('../dataset/pos_tok.pkl', 'rb'))
neg_tok = pickle.load(open('../dataset/neg_tok.pkl', 'rb'))

input_len = 100
pad_token = '_pad_'
unk_token = '_unk_'

def sub_space(tok_sentence):
	return list(map(lambda token: '_space_' if token == ' ' else token, tok_sentence))
def sub_lol(tok_sentence):
	return list(map(lambda token: '_lol_' if token == 'lol' else token, tok_sentence))
def pad_sentence(tok_sentence):
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
np.save('../dataset/pos_int.npy', pos_int)
np.save('../dataset/neg_int.npy', neg_int)
print(unknown_words, len(unknown_words))
pickle.dump(unknown_words, open('../dataset/unknown.pkl', 'wb'))