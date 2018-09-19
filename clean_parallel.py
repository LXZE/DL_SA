import multiprocessing as mp
import sys
import csv
import re
import pandas as pd
import numpy as np
import pythainlp as pyt
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
# Pattern
user_pattern = re.compile(r'@(\w){1,15}\s')
emoji_pattern = re.compile("["
	u"\U0001F600-\U0001F64F"  # emoticons
	u"\U0001F300-\U0001F5FF"  # symbols & pictographs
	u"\U0001F680-\U0001F6FF"  # transport & map symbols
	u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
						"]+", flags=re.UNICODE)
multichar_emoji_pattern = re.compile(r'(<3)')
url_pattern = re.compile(r'(https:\S+)')
hashtag_pattern = re.compile(r'(\#\S+)')

duplicate_space = re.compile(r'(\s{2,})')

special_char = re.compile(r'&\S+;')
non_char = re.compile(r'"|\'|\\|/|!|_|-|—|=|\+|\.|\n|\(|\)|\*|•|@|\?|\^|~|“|”|\[|\]|{|}|<|>|:|;|\|')

def cleanLine(line):
	# pattern create and remove space
	line = re.sub(user_pattern, ' ', line)

	# clear all unneccessary text
	line = re.sub(emoji_pattern, '', line)
	line = re.sub(multichar_emoji_pattern, '', line)
	line = re.sub(url_pattern, '', line)
	line = re.sub(hashtag_pattern, '', line)
	line = re.sub(special_char, '', line)
	line = re.sub(non_char, '', line)

	line = re.sub(duplicate_space, ' ', line)

	return line.replace('\n',' ')

non_thai_eng_pattern = re.compile(r'([^\u0E00-\u0E7Fa-zA-Z0-9฿%\s]+)')
def filtering(line):
	non_thai_eng_pattern = re.compile(r'([^\u0E00-\u0E7Fa-zA-Z0-9#฿%\s]+)')
	return re.sub(non_thai_eng_pattern, '', line)

def stripping(line):
	tmp = line.lstrip(' ')
	return tmp.rstrip(' ')

lol_pattern = re.compile(r'(5{2,}\+?)')
vowel_error = re.compile(r'เเ')
repeat_pattern = re.compile(r'(\S+?)\1+')
repeatable_char = ['อ', 'ร', 'ว', 'ย']
# line_sub = lambda line, match, count: re.sub('{}'.format(match.group(0)), match.group(1)*count, line, count=1)
def fixing(line):
	line = re.sub(lol_pattern, 'lol', line)
	line = re.sub(vowel_error, 'แ', line)
	line = re.sub(repeat_pattern, r'\1\1', line)
	'''
	try:
		if bool(repeat_pattern.search(line)):
			for match in repeat_pattern.finditer(line):
				# if match.group(1) in repeatable_char: # if match char set in repeatable list
				# 	pos = re.search(match.group(0),line).start()
				# 	if line[pos-1] in ['เ', 'แ'] or match.group(1) == 'ร': # if char before that repeat char is e|er-vowel or it's 'r' then twice
				# 		line = line_sub(line, match, 2)
				# 	else: # if char before not match condition then it should be once
				# 		line = line_sub(line, match, 1)
				# else: # if match char set not in repeatable char
				# 	if len(match.group(1)) == 1: # if not repeat char but match len = 1 so repeat once
				# 		line = line_sub(line, match, 1)
				# 	else: # if repeat char is kind of pattern then twice
				# 		line = line_sub(line, match, 2)
				line = line_sub(line, match, 2)
		else:
			pass
	except AttributeError:
		pass
	'''
	line = ''.join(list(filter(lambda word: len(word) > 1 or word == ' ', pyt.word_tokenize(line, engine='newmm'))))
	return line

try:
	file = open(sys.argv[1],'r')
	fileName = sys.argv[1].split('/')[-1].split('.')[0]
except IndexError:
	print('Please give the input file')
	exit(0)
except FileNotFoundError:
	print('File not found')
	exit(0)

pattern = re.compile('\[\d{4}/\d{2}/\d{2}-(?:\d{1,2}:){2}\d{1,2}\],\n')

def process(lines, core, acc_result):
	tableData = pd.DataFrame(columns=['time','text'])
	tmp = ''
	i=0
	for line in lines:
		if bool(pattern.search(line)):
			result = stripping(tmp.replace('\ufeff',''))
			result = filtering(result)
			if len(result) == 0:
				tmp = ''
				continue

			tableData = tableData.append({
				'time': line[1:-3],
				'text': stripping(fixing(result))
			}, ignore_index=True)
			tmp = ''
			i+=1
		else:
			tmp += cleanLine(line)
	acc_result.append((core, tableData))
	print('core {} finished'.format(core))

if __name__ == '__main__':
	print('start manipulate')
	ncore = mp.cpu_count()
	# output = mp.Queue()
	output_list = mp.Manager().list()

	all_line = file.readlines()
	line_chunks = np.array_split(all_line, ncore)
	processes = [mp.Process(target=process, args=(line_chunks[core], core, output_list)) for core in range(ncore)]

	print('start process')
	for p in processes:
		p.start()

	for p in processes:
		p.join()

	print('start concat')
	print(list(map(lambda x: x[0], output_list)))
	results_tup = sorted(output_list, key=lambda x: x[0])
	result_table = pd.concat(list(map(lambda data: data[1], results_tup)))

	print('start write file')
	result_table.to_csv('data/clean/'+fileName+'_clean.csv', index=False,
					quoting=csv.QUOTE_NONNUMERIC, doublequote=False, escapechar="\\")
	# TODO: print unwant char into some txt file, for further filtering
	# pp.pprint(all_unwant)
