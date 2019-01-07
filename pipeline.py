import glob
from subprocess import call
import pathlib
import time

getTime = lambda : time.strftime("%Y-%m-%d %H:%M:%S")

# read and clean
rawFileList = sorted(glob.glob('data/twit_*'))
print(rawFileList)
pathlib.Path('data/clean').mkdir(parents=True, exist_ok=True)
for file in rawFileList:
	print('[{}] Cleaning {}'.format(getTime(), file))
	code = call(['python', 'twit_reader.py', file])
	print('[{}] Finish cleaning {} with code {}'.format(getTime(), file, code))
	print('-'*20)

# filter
cleanFileList = sorted(glob.glob('data/clean/*'))
print(cleanFileList)
pathlib.Path('data/filtered').mkdir(parents=True, exist_ok=True)
for file in cleanFileList:
	print('[{}] Filtering {}'.format(getTime(), file))
	code = call(['python', 'filter.py', file, 'sentiment.txt'])
	print('[{}] Finish filtering {} with code {}'.format(getTime(), file, code))
	print('-'*20)

# gather
code = call(['python', 'gather_data.py', 'sentiment.txt'])