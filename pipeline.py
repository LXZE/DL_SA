import glob
from subprocess import call
import pathlib

# clean
rawFileList = glob.glob('data/twit_*')
print(rawFileList)
pathlib.Path('data/clean').mkdir(parents=True, exist_ok=True) 
for file in rawFileList:
	code = call(['python', 'clean.py', file])
	print(code)

# filter
cleanFileList = glob.glob('data/clean/*')
print(cleanFileList)
pathlib.Path('data/filtered').mkdir(parents=True, exist_ok=True) 
for file in cleanFileList:
	code = call(['python', 'filter.py', file])

# gather
code = call(['python', 'gather_data.py', file])
