import glob

entity = 'dtac'
fileList = glob.glob('data/filtered/{}*.txt'.format(entity))
print(fileList)

for filePath in fileList:
	file = open(filePath, 'r')
	print(file.readlines())
	