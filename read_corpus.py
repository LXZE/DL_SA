import hashlib
import pandas as pd
import glob
import csv

data = pd.DataFrame(columns=['class', 'message'])
hashlist = []
idx = 0
filelist = sorted(glob.glob('./sentiment_analysis_thai/corpus/*.csv'))
for filename in filelist:
	file = csv.reader(open(filename, 'r'), delimiter=',')
	print(filename)
	next(file, None)
	for row in file:
		hash_val = hashlib.sha256(row[1].encode('utf8')).hexdigest()
		if hash_val not in hashlist:
			data.loc[idx] = [row[0], row[1]]
			idx+=1
			hashlist.append(hash_val)
print(idx)

# TODO: Clean and save result to text, split by class too