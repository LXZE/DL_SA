import sys
sys.path.append('./thai-word-segmentation')
from thainlplib import ThaiWordSegmentLabeller
import tensorflow as tf
import pandas as pd

saved_model_path='./thai-word-segmentation/saved_model'

def nonzero(a):
    return [i for i, e in enumerate(a) if e != 0]

def split(s, indices):
    return [s[i:j] for i,j in zip(indices, indices[1:]+[None])]	

try:
	dataset = pd.read_csv(sys.argv[1], escapechar="\\")
except IndexError:
	print('Please give the input file')
	exit(0)
except FileNotFoundError:
	print('File not found')
	exit(0)

with tf.Session() as session:
	model = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], saved_model_path)
	signature = model.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
	graph = tf.get_default_graph()

	g_inputs = graph.get_tensor_by_name(signature.inputs['inputs'].name)
	g_lengths = graph.get_tensor_by_name(signature.inputs['lengths'].name)
	g_training = graph.get_tensor_by_name(signature.inputs['training'].name)
	g_outputs = graph.get_tensor_by_name(signature.outputs['outputs'].name)

	for idx,row in dataset.iterrows():
		print(idx)
		text = row['text']
		inputs = [ThaiWordSegmentLabeller.get_input_labels(text)]
		lengths = [len(text)]

		y = session.run(g_outputs, feed_dict = {g_inputs: inputs, g_lengths: lengths, g_training: False})
		for w in split(text, nonzero(y)): print(w, end='|')
		# print()
		print('-'*20)


# for idx,row in dataset.iterrows():
# 	# print(row)
# 	inputs = [ThaiWordSegmentLabeller.get_input_labels(row['text'])]
# 	lengths = [len(text)]
	# print(row['text'], inputs)
	# print('-'*20)
	# if idx == 20:
	# 	break