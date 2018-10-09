import re, sys
import pandas as pd
import pythainlp as pyt
try:
	# NOTE: input must be cleaned
	dataset = pd.read_csv(sys.argv[1], escapechar="\\")
except IndexError:
	print('Please give the input file')
	exit(0)
except FileNotFoundError:
	print('File not found')
	exit(0)

prop = [0, 0]
thai_pattern = re.compile(r'([\u0E00-\u0E7F฿%]+)')
rep = ['งง', 'สส']
for idx, row in dataset.iterrows():
	sentence = row['text']
	result = pyt.word_tokenize(sentence, engine='newmm')
	if any([len(token) == 1 and bool(thai_pattern.search(token)) for token in result]):
		# print(result)
		prop[0] += 1
	elif any([len(token) == 2 and bool(thai_pattern.search(token)) and token[0] == token[1] and (token not in rep) for token in result]):
		prop[0] += 1
		print(result)
	else:
		prop[1] += 1
print(prop)