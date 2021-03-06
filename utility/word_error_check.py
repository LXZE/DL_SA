# -*- coding: utf-8 -*-
# TODO: store word list with pickle and accumulate with other file later
import multiprocessing as mp
from collections import defaultdict as ddict
from itertools import chain
import re, sys, os
import pandas as pd
import numpy as np
import pythainlp as pyt
import clean
import time

thai_pattern = re.compile(r'([\u0E00-\u0E7F฿%]+)')
char_repeat_pattern = re.compile(r'([^\d\s]{1})\1+')
def process(lines, core, acc_result):
	print('core {} started'.format(core))
	pattern_count = {}
	for idx, line in enumerate(lines):
		sentence = clean.fixing(line)
		result = pyt.word_tokenize(sentence, engine='newmm')

		crit0 = [bool(char_repeat_pattern.search(token)) and bool(thai_pattern.search(token)) for token in result]
		res0 = [(result[idx-1: idx+1]) for idx, log in enumerate(crit0) if log]
		if any(crit0):
			for patt in res0:
				if len(patt) == 0 or patt[0] == ' ':
					continue
				patt = '+'.join(patt)
				if patt in pattern_count:
					pattern_count[patt] += 1
				else:
					pattern_count[patt] = 1

	acc_result.append((core, pattern_count))
	print('core {} finished'.format(core))

if __name__ == '__main__':
	try:
		# NOTE: input must be cleaned
		dataset = pd.read_csv(sys.argv[1], escapechar="\\", encoding='utf8')
	except IndexError:
		print('Please give the input file')
		exit(0)
	except FileNotFoundError:
		print('File not found')
		exit(0)

	ncore = mp.cpu_count()
	output_list = mp.Manager().list()

	all_line = dataset['text'].tolist()
	line_chunks = np.array_split(all_line, ncore)
	processes = [mp.Process(target=process, args=(line_chunks[core], core, output_list)) for core in range(ncore)]

	for i, p in enumerate(processes):
		p.start()

	# if os.name == 'nt':
	# 	# MP is likely to broke when running in windows, so I put this to stop the hang process.
	#   # After reproduce a several attempt on data distribution, let process consume more ram make this works.
	#   # I don't know why.
	# 	while True:
	# 		time.sleep(1)
	# 		alive = [(i, p.pid) for i, p in enumerate(processes) if p.is_alive()]
	# 		if alive:
	# 			print(len(alive), 'processes alive; among them:', alive)
	# 			if len(alive) == 1:
	# 				pass
	# 				# processes[alive[0][0]].terminate()
	# 				# break
	# 		else:
	# 			print('no process alive now')
	# 			break

	for p in processes:
		p.join()

	print('interpret accumulate result')
	print(list(map(lambda x: x[0], output_list)))
	res_dict = ddict(int)
	for k,v in chain.from_iterable([res.items() for _, res in output_list]):
		res_dict[k] += v
	patt_dict = [(k, res_dict[k]) for k in sorted(res_dict, key=res_dict.get, reverse=True)]
	file = open('error.txt', 'w', encoding='utf8')
	for k,v in patt_dict:
		file.write('{} {}\n'.format(k,v))
	file.close()

'''
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
'''