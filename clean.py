# -*- coding: utf-8 -*-

import sys, csv, re
import pandas as pd
import numpy as np
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
# Pattern
user_pattern = re.compile(r'@(\w){1,15}\s')
emoji_pattern = re.compile("["
	u"\U0001F600-\U0001F64F"  # emoticons
	u"\U0001F300-\U0001F5FF"  # symbols & pictographs
	u"\U0001F680-\U0001F6FF"  # transport & map symbols
	u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
	u"\U00002702-\U000027B0"
	u"\U000024C2-\U0001F251"
	"]+", flags=re.UNICODE)
multichar_emoji_pattern = re.compile(r'(<3)')
url_pattern = re.compile(r'(https:\S+)')
hashtag_pattern = re.compile(r'(\#\S+)')

duplicate_space = re.compile(r'(\s{2,})')

special_char = re.compile(r'&\S+;')
non_char = re.compile(r'"|\'|\\|/|!|_|-|—|=|\+|\.|\n|\(|\)|\*|&|•|@|\?|\^|~|“|”|\[|\]|{|}|<|>|:|;|\|')

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
	return re.sub(non_thai_eng_pattern, '', line)

# removing whitespace
def stripping(line):
	tmp = re.sub('\s+',' ',line)
	tmp = tmp.lstrip(' ')
	return tmp.rstrip(' ')

# fixing, lower, replacing any kind of pattern and error in Thai
lol_pattern = re.compile(r'([56]{2,}\+?)|(ถ{2,}\+?)|(๕{2,}\+?)')
vowel_error = re.compile(r'เเ')
repeat_pattern = re.compile(r'([^\d\s]+?)\1+')
def fixing(line):
	line = re.sub(lol_pattern, ' lol ', line)
	line = re.sub('ๆ+', 'ๆ', line)
	line = re.sub(duplicate_space, ' ', line)
	line = re.sub(vowel_error, 'แ', line)
	line = min_char(line)
	return line.lower()

file = open('./utility/resource/thaiword.txt', 'r', encoding='utf8')
word_dict = map(lambda line: line[:-1] ,file.readlines())
word_sub = lambda line, match, count: re.sub('{}'.format(match.group(0)), match.group(1)*count, line, count=1)
def clean_word(word):
	res_word = word
	try:
		if bool(repeat_pattern.search(word)):
			if word in word_dict:
				return res_word
			for match in repeat_pattern.finditer(word):
				res_word = word_sub(res_word, match, 2)
		else:
			pass
	except AttributeError:
		pass
	return res_word

thai_vowel = 'ะัาำิีึืุูเแโใไ็่้๊๋์'
# NOTE: ก 0e01 ๙ 0e59 โคมูตร จาถิ 0e5b
# be able to use unicode char inside bracket directly
# for example [ะ-ไ] [เ-ไ]
thai_pattern = re.compile(r'([\u0E00-\u0E7F฿%]+)')
char_repeat_pattern = re.compile(r'([^\d\s]{1})\1+')
def min_char(line):
	# 2,3 might be word, else it would be intensifying
	# if char repeat == vowel then sub to 1
	try:
		if bool(char_repeat_pattern.search(line)):
			for match in char_repeat_pattern.finditer(line):
				if match.group(1) in thai_vowel:
					line = word_sub(line, match, 1)
				elif len(match.group(0)) in [2,3]:
					pass
				else:
					line = word_sub(line, match, 2)
	except AttributeError:
		pass
	return line