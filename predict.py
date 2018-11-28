import numpy as np
from collections import defaultdict as ddict
from keras.models import load_model
from gensim.models import KeyedVectors
import pythainlp as pyt
import clean
from attention import AttentionWithContext as att

vector_model_dir = '../model/thai2vec/word2vec/' 
vector_model_path_bin = f'{vector_model_dir}thai2vec02.bin'
vector_model = KeyedVectors.load_word2vec_format(vector_model_path_bin, binary=True)

itos = vector_model.index2word
itos.insert(0, '_lol_')
stoi = ddict(lambda: 0, {v:k for k,v in enumerate(itos)})

model = load_model('../model/model.hdf5', custom_objects={'AttentionWithContext': att})
print("Loaded model from disk")

predict_txt = 'ถ้า dtac ยังแก้ปัญหาให้ดิฉันไม่ได้ ดิฉันก็จะยุให้คนรอบข้างย้ายออกให้หมด ตอนนี้ก็ 3 คนแล้วอ่ะค่ะ ถ้ายังแก้ไม่ได้คนก็จะออกเพิ่มเรื่อย ๆ ไล่ฆ่าแบบแก๊งร่วมอาสา อย่าเพิ่งรีบตาย ชั้นเพิ่งจะเริ่ม'

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
def sen2int(tok_list):
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
				tmp.append(stoi[unk_token])
		new_int_sentence.append(tmp)
	return new_int_sentence

token = clean.fixing(predict_txt)
token = pyt.word_tokenize(token, engine='newmm')
token = list(map(lambda x: clean.clean_word(x), token))
print(token)
token = np.array(sen2int([token]))
# print(token.shape)

res = ['negative', 'positive']
debug = model.predict(token, batch_size=1)[0]
res_y = model.predict_classes(token, batch_size=1)
print('Data: {}'.format(predict_txt))
target = res[res_y[0]]
print('Result: {}'.format(target))
print('Confidential: Negative={:2.4f}% / Positive={:2.4f}%'.format(float(debug[0])*100, float(debug[1])*100))
