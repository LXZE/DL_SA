import re
repeat_pattern = re.compile(r'([^\d\s]+?)\1+')
file = open('./utility/resource/thaiword.txt', 'r', encoding='utf8')
pattern = set()
for lines in file.readlines():
	word = lines[:-1]
	if bool(repeat_pattern.search(word)):

		for match in repeat_pattern.finditer(word):
			g = match.group()
			if(len(g) > 2):
				print(word, end=', ')
				print(g, len(g), end=', ')
				print()