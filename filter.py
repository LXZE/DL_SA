from pprint import PrettyPrinter
import sys, os
import pandas as pd

pp = PrettyPrinter(indent=4)

try:
	dataset = pd.read_csv(sys.argv[1], escapechar="\\", encoding='utf8')
	if os.name == 'nt':
		fileName = sys.argv[1].split('\\')[-1].split('.')[0]
	else:
		fileName = sys.argv[1].split('/')[-1].split('.')[0]
except IndexError:
	print('Please give the input file')
	exit(0)
except FileNotFoundError:
	print('Input file not found')
	exit(0)

try:
	file = open(sys.argv[2],'r', encoding='utf8')
except IndexError:
	print('Please give the entity file')
	exit(0)
except FileNotFoundError:
	print('Entity file not found')
	exit(0)

entities = {}
for line in file.readlines():
	if line[-1] == '\n':
		line = line[:-1]
	tmp = line.split(',')
	entities[tmp[0]] = {}
	entities[tmp[0]]['filter'] = tmp
	entities[tmp[0]]['txt'] = []

for idx,row in dataset.iterrows():
	text = row['text']
	for label, entity in entities.items():
		if any(substr in text for substr in entity['filter']):
			entities[label]['txt'].append(text)
# print(entities)
# pp.pprint(entities)
for label, entity in entities.items():
	file = open('data/filtered/{}-{}.txt'.format(label,fileName),'w', encoding='utf8')
	for line in entity['txt']:
		file.write(line)
		file.write('\n')
	file.close()