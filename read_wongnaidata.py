import pandas as pd
import re

dataset_path = '../dataset/wongnai/w_review_train.csv'
df = pd.read_csv(dataset_path, delimiter=';', header=None, 
	names=['text','label'], index_col=False)

df_seperated = []


for i in range(1,6):
	tmp = pd.DataFrame(df[df['label'] == i]).reset_index()
	df_seperated.append(tmp['text'])


for idx, df_elem in enumerate(df_seperated):
	df_seperated[idx] = df_elem.apply(lambda x: x.replace('\n', ' '))

df_seperated[0].to_csv('../dataset/wongnai_1.csv', sep=';')
df_seperated[4].to_csv('../dataset/wongnai_5.csv', sep=';')