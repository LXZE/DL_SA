import glob
import pandas as pd
from subprocess import call
import time

getTime = lambda : time.strftime("%Y-%m-%d %H:%M:%S")

fileList = sorted(glob.glob('data/clean/twit_*'))
file = []
for fileName in fileList[:3]:
	print(fileName)
	tmp = pd.read_csv(fileName)
	file.append(tmp)
print(file)
