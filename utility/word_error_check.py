import re, sys
import pandas as pd
import pythainlp as pyt
import clean
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
# repeat_pattern = re.compile(r'([^\d\s]+?)\1+')
char_repeat_pattern = re.compile(r'([^\d\s]{1})\1+')
rep = ['งง', 'สส', 'นน', 'รร', 'ออ', 'บบ']

pattern_count = {}

for idx, row in dataset.iterrows():
	sentence = clean.fixing(row['text'])
	result = pyt.word_tokenize(sentence, engine='newmm')

	crit0 = [bool(char_repeat_pattern.search(token)) and bool(thai_pattern.search(token)) for token in result]
	res0 = [(result[idx-1: idx+1]) for idx, log in enumerate(crit0) if log]

	crit2 = [len(token) == 2 and bool(thai_pattern.search(token)) and token[0] == token[1] and (token not in rep) for idx, token in enumerate(result)]
	res2 = [(result[idx-1: idx+1]) for idx, log in enumerate(crit2) if log]

	if any(crit0):
		# print(res0)
		for patt in res0:
			if len(patt) == 0 or patt[0] == ' ':
				continue
			patt = '+'.join(patt)
			if patt in pattern_count:
				pattern_count[patt] += 1 
			else:
				pattern_count[patt] = 1

	# if any([len(token) == 1 and bool(thai_pattern.search(token)) for token in result]):
	# 	prop[0] += 1
	# elif any(crit2):
	# 	prop[0] += 1
	# 	print(res2)
	# else:
	# 	prop[1] += 1
patt_dict = [(k, pattern_count[k]) for k in sorted(pattern_count, key=pattern_count.get, reverse=True)]
for k,v in patt_dict:
	print(k, v)
# print(prop)