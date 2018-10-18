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
rep = ['งง', 'สส', 'นน', 'รร', 'ออ']
for idx, row in dataset.iterrows():
	sentence = row['text']
	result = pyt.word_tokenize(sentence, engine='newmm')
	crit2 = [len(token) == 2 and bool(thai_pattern.search(token)) and token[0] == token[1] and (token not in rep) for idx, token in enumerate(result)]
	res2 = [(result[idx-1: idx+1]) for idx, log in enumerate(crit2) if log]
	if any([len(token) == 1 and bool(thai_pattern.search(token)) for token in result]):
		prop[0] += 1
	elif any(crit2):
		prop[0] += 1
		print(res2)
	else:
		prop[1] += 1
print(prop)