import pythainlp as pyt
import pandas as pd
from tqdm import tqdm
result = pd.read_csv('./result_eval.csv', index_col=0)
result = result.astype(str)



tmp = []
types = list(result.columns.values)[2:]
for idx, row in result.iterrows():
	for t in types:
		if row[t] == row['target']:
			result.at[idx, t] = '*'
			tmp.append(t)
		else: result.at[idx, t] = ''



	if len(tmp) == len(types) or len(tmp) == 0:
		result.drop(idx, inplace=True)
	tmp = []


result.to_excel('new_result.xlsx')

# types = list(result.columns.values)[2:]
# select_idx = [0,1,5,6]
# filtered = [[] for i in range(len(select_idx))]

# alles = []

# tmp = []
# for idx, row in result.iterrows():
# 	for t in types:
# 		if row['target'] == row[t]:
# 			tmp.append(t)

# 	# and
# 	for j, idx in enumerate(select_idx):
# 		if len(tmp) == 1 and types[idx] in tmp:
# 			filtered[j].append(row)

# 	# or
# 	# for j, idx in enumerate(select_idx):
# 	# 	if len(tmp) < len(types) and types[idx] in tmp:
# 	# 		filtered[j].append(row)

# 	if len(tmp) == len(types):
# 		alles.append(row)

# 	tmp = []

# for j, idx in enumerate(select_idx):
# 	filtered[j] = pd.DataFrame(filtered[j])
# 	filtered[j] = filtered[j][['target', 'text']]
# 	filtered[j].to_csv(f'result/{types[idx]}.csv')
# '''
# alles = pd.DataFrame(alles)
# alles = alles[['target', 'text']]
# # alles.to_csv('result/all.csv')

# result = []
# ans = {'pos':1, 'neg':0}
# pbar = tqdm(total=len(alles))
# for idx, row in alles.iterrows():
# 	res = pyt.sentiment(row['text'])
# 	res = ans[res]
# 	if res != row['target']:
# 		# print(row['text'])
# 		result.append(row['text'])
# 	pbar.update(1)
# pbar.close()
# result = pd.DataFrame(result)
# result.to_csv('result/all_not_nb.csv')'''