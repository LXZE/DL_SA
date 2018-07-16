import sys
import csv
import re
import pandas as pd

def cleanLine(line):
	return line.replace('\n',' ')

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
		tableData = tableData.append({
			'time':line[1:-3],
			'text':tmp.replace('\ufeff','')
		}, ignore_index=True)
		tmp = ''
		if '"' in tmp:
			print('found quote at ',i)
		# print(i)
		i+=1
	else:
		tmp += cleanLine(line)

tableData.to_csv(fileName+'_clean.csv', index=False,
				quoting=csv.QUOTE_NONNUMERIC, doublequote=False, escapechar="\\")