from keras.models import model_from_json
from keras.models import load_model

from gensim.models import KeyedVectors
import numpy as np

pos = np.load('../dataset/wn_pos_int.npy')
neg = np.load('../dataset/wn_neg_int.npy')

x = np.concatenate((pos,neg), axis = 0)
y = np.concatenate(
	(
		np.full((pos.shape[0], 2), [0,1]),
		np.full((neg.shape[0], 2), [1,0])
	), axis = 0
)

vector_model_dir = '../model/thai2vec/word2vec/' 
vector_model_path_bin = f'{vector_model_dir}thai2vec02.bin'
vector_model = KeyedVectors.load_word2vec_format(vector_model_path_bin, binary=True)

# word_dict = {}
# word_dict['_lol_'] = vector_model['555']
# for word in vector_model.index2word:
# 	word_dict[word] = vector_model[word]
# word_vec = pd.DataFrame.from_dict(word_dict, orient='index')

itos = vector_model.index2word
itos.insert(0, '_lol_')

json_file = open('../model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("../model/model.h5")
# model.save('model_num.hdf5')
# loaded_model=load_model('model_num.hdf5')
print("Loaded model from disk")

def sub_space_tok(tok_sentence):
	return list(map(lambda token: ' ' if token == '_space_' else token, tok_sentence))

rand = np.random.choice(len(x), 1, replace=False)
res = ['bad', 'good']
for i in rand:
	pre_x, pre_y = x[i], y[i]
	pre_x = pre_x.reshape(1, 600)
	debug = loaded_model.predict(pre_x, batch_size=1)[0]
	res_y = loaded_model.predict_classes(pre_x, batch_size=1)
	tmp = list(filter(lambda elem: elem != 2, pre_x[0]))
	tmp = list(map(lambda elem: itos[elem], tmp))
	tmp = sub_space_tok(tmp)
	sentence = ''.join(tmp)
	print('Data: {}'.format(sentence))
	target = res[res_y[0]]
	prediction = res[int(str(np.where(pre_y==1.)[0])[1])]

	print('Prediction: {}'.format(prediction))
	print('Result: {}'.format(target))
	print('Confidential: bad={:2.4f}% / good={:2.4f}%'.format(float(debug[0])*100, float(debug[1])*100))
