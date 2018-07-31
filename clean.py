import sys
import csv
import re
import pandas as pd
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

def cleanLine(line):
	user_pattern = re.compile(r'@(\w){1,15}\s')
	emoji_pattern = re.compile("["
		u"\U0001F600-\U0001F64F"  # emoticons
		u"\U0001F300-\U0001F5FF"  # symbols & pictographs
		u"\U0001F680-\U0001F6FF"  # transport & map symbols
		u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
						   "]+", flags=re.UNICODE)
	url_pattern = re.compile(r'(https:\S+)')
	hashtag_pattern = re.compile(r'(\#\S+)')

	duplicate_space = re.compile(r'(\s{2,})')

	non_char = re.compile(r'"|\'|\\|/|!|_|-|—|=|\+|\.|\n|\(|\)|\*|•|@|\?|\^|~|“|”|\[|\]|{|}|:|\|')
	special_char = re.compile(r'&\S+;')

	# pattern create and remove space
	line = re.sub(user_pattern, ' ', line)

	# clear all unneccessary text
	line = re.sub(emoji_pattern, '', line)
	line = re.sub(url_pattern, '', line)
	line = re.sub(hashtag_pattern, '', line)
	line = re.sub(non_char, '', line)
	line = re.sub(special_char, '', line)

	line = re.sub(duplicate_space, ' ', line)

	return line.replace('\n',' ')

def filtering(line):
	tmp = set()
	non_thai_eng_pattern = re.compile(r'([^\u0E00-\u0E7Fa-zA-Z0-9฿%\s]+)')
	non_char_set = set(re.findall(non_thai_eng_pattern, line))
	for n in non_char_set:
		tmp.update(n)
	return [re.sub(non_thai_eng_pattern, '', line), tmp]

def stripping(line):
	tmp = line.lstrip(' ')
	return tmp.rstrip(' ')

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
tableData = pd.DataFrame(columns=['time','text'])

all_unwant = set()
tmp = ''
i=0
for line in file.readlines():
	if bool(pattern.search(line)):
		result = stripping(tmp.replace('\ufeff',''))
		[result, unwant] = filtering(result)
		all_unwant.update(unwant)
		if len(result) == 0:
			tmp = ''
			continue

		tableData = tableData.append({
			'time': line[1:-3],
			'text': result
		}, ignore_index=True)
		tmp = ''
		# if '"' in tmp:
		# 	print('found quote at ',i)
		# print(i)
		i+=1
	else:
		tmp += cleanLine(line)

tableData.to_csv('data/clean/'+fileName+'_clean.csv', index=False,
				quoting=csv.QUOTE_NONNUMERIC, doublequote=False, escapechar="\\")
# TODO: print unwant char into some txt file, for further filtering
pp.pprint(all_unwant)