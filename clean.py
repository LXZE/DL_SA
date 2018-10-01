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

# removing username, emoji, url, hashtag, or anykind of special character
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

# filtering only thai, numerical and latin character
non_thai_eng_pattern = re.compile(r'([^\u0E00-\u0E7Fa-zA-Z0-9฿%\s]+)')
def filtering(line):
	non_thai_eng_pattern = re.compile(r'([^\u0E00-\u0E7Fa-zA-Z0-9#฿%\s]+)')
	return re.sub(non_thai_eng_pattern, '', line)

# removing whitespace
def stripping(line):
	tmp = line.lstrip(' ')
	return tmp.rstrip(' ')

# fixing, replacing any kind of pattern and error in Thai
lol_pattern = re.compile(r'(5{2,}\+?)')
vowel_error = re.compile(r'เเ')
repeat_pattern = re.compile(r'([^\d\s]+?)\1+')
def fixing(line):
	line = re.sub(lol_pattern, 'lol', line)
	line = re.sub(vowel_error, 'แ', line)
	line = re.sub(repeat_pattern, r'\1\1', line)
	return line

word_sub = lambda line, match, count: re.sub('{}'.format(match.group(0)), match.group(1)*count, line, count=1)
def clean_word(word):
	try:
		if bool(repeat_pattern.search(word)):
			for match in repeat_pattern.finditer(word):
				word = word_sub(word, match, 2)
		else:
			pass
	except AttributeError:
		pass
	return word