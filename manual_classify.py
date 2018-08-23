import glob
import sys, os
import pathlib
from pythainlp import sentiment as sent


dataDirectory = '../dataset'
pathlib.Path(dataDirectory).mkdir(parents=True, exist_ok=True)

try:
	inputFileName = sys.argv[1]
	inputFile = open(inputFileName, 'r')
except IndexError:
	print('Please give the input file')
	exit(0)
except FileNotFoundError:
	print('File not found')
	exit(0)

try:
	fileName = [
		'/pos_predict.txt',
		'/neg_predict.txt'
	]
	fileName = list(map(lambda x: dataDirectory+x,fileName))
	for path in fileName:
		if not os.path.isfile(path):
			pathlib.Path(path).touch()
	posFile = open(fileName[0], 'a')
	negFile = open(fileName[1], 'a')

except Exception as e:
	print('Error')
	print(e)
	exit(0)

pos = []
neg = []
for lineNum, line in enumerate(inputFile.readlines()):
	res = sent(line)
	if res == 'pos':
		pos.append(line)
	elif res == 'neg':
		neg.append(line)

def writeToFile(file, item):
	for line in item:
		file.write(line)
writeToFile(posFile, pos)
writeToFile(negFile, neg)
posFile.close()
negFile.close()

inputFile.close()