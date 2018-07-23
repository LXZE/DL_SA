import sys
import csv
import re
import pandas as pd

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

def stripping(line):
	return line.lstrip(' ').rstrip(' ')

try:
	file = open(sys.argv[1],'r')
	fileName = sys.argv[1].split('.')[0]
except IndexError:
	print('Please give the input file')
	exit(0)
except FileNotFoundError:
	print('File not found')
	exit(0)

pattern = re.compile('\[\d{4}/\d{2}/\d{2}-(?:\d{1,2}:){2}\d{1,2}\],\n')
tableData = pd.DataFrame(columns=['time','text'])

tmp = ''
i=0
for line in file.readlines():
	if bool(pattern.search(line)):
		result = stripping(tmp.replace('\ufeff',''))
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

tableData.to_csv(fileName+'_clean.csv', index=False,
				quoting=csv.QUOTE_NONNUMERIC, doublequote=False, escapechar="\\")