import glob, sys

try:
	entityFile = open(sys.argv[1],'r', encoding='utf8')
except IndexError:
	print('Please give the entity file')
	exit(0)
except FileNotFoundError:
	print('Entity file not found')
	exit(0)

try:
	readed = {}
	entities = {}
	for line in entityFile.readlines():
		if line[-1] == '\n':
			line = line[:-1]
		tmp = line.split(',')
		entities[tmp[0]] = {}
		entities[tmp[0]]['filter'] = tmp
		entities[tmp[0]]['txt'] = []

	for entity in entities.keys():
		readed[entity] = []
		fileList = glob.glob('data/filtered/{}*.txt'.format(entity))
		for filePath in fileList:
			file = open(filePath, 'r', encoding='utf8')
			readed[entity] += file.readlines()

except FileNotFoundError:
	print('File not found')
	exit(0)

# print(readed)
for key, content in readed.items():
	print(key)
	file = open('data/filtered/{}.txt'.format(key),'w', encoding='utf8')
	for line in content:
		file.write(line)
	file.close()