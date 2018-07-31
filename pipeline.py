import glob
from subprocess import call

# clean
rawFileList = glob.glob('data/twit_*')
print(rawFileList)
for file in rawFileList:
	code = call(['python', 'clean.py', file])
	print(code)

# filter
cleanFileList = glob.glob('data/clean/*')
print(cleanFileList)
for file in cleanFileList:
	code = call(['python', 'filter.py', file])

# gather
code = call(['python', 'gather_data.py', file])
