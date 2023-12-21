import pandas as pd
import numpy as np
import os
from skimage import io

meta_path = '../../Downloads/Datasets-main/basket_volley/'
classes = os.listdir(meta_path)
paths = []
for c in classes:
	sub_classes = os.listdir(os.path.join(meta_path, c))
	for s_c in sub_classes:
		ims = os.listdir(os.path.join(meta_path, c, s_c))
		for im in ims:
			paths.append(os.path.join(meta_path, c, s_c, im))

cols = ['Path', 'Target', 'Color', 'Gender', 'Train']
df = pd.DataFrame(columns=cols)

ids = np.arange(len(paths))
np.random.shuffle(ids)
split_index = int(len(paths)*0.2)
train_ids = ids[split_index:]
test_ids = ids[:split_index]

to_num = {
	'basket': 0,
	'volley': 1,
	'm': 0,
	'f': 1,
	'r': 0,
	'y': 1
}

for i in train_ids:
	path = paths[i]
	target, gender, color = path.split('/')[-2].split('_')
	df.loc[len(df.index)] = [path, to_num[target],to_num[color],to_num[gender], True]
		
for i in test_ids:
	path = paths[i]
	target, gender, color = path.split('/')[-2].split('_')
	df.loc[len(df.index)] = [path, to_num[target],to_num[color],to_num[gender], False]

df.to_csv(os.path.join(meta_path, 'splits.csv'))

print(df[df['Train'] == True]['Target'].value_counts())
print(df[df['Train'] == False]['Target'].value_counts())

print(df[df['Train'] == True]['Color'].value_counts())
print(df[df['Train'] == False]['Color'].value_counts())

print(df[df['Train'] == True]['Gender'].value_counts())
print(df[df['Train'] == False]['Gender'].value_counts())

"""
df = pd.read_csv(os.path.join(meta_path,'HAM10000_metadata.csv'))
ids = df['lesion_id'].unique()
np.random.shuffle(ids)

split_index = int(0.2*len(ids))
ids_train = ids[split_index:]
ids_test = ids[:split_index]

train_files = []
for id_train in ids_train:
	train_files.extend(df[df['lesion_id'] == id_train].index)

test_files = []
for id_test in ids_test:
	test_files.extend(df[df['lesion_id'] == id_test].index)

train_files = np.array(train_files)
test_files = np.array(test_files)

np.save(os.path.join(meta_path, 'train_ids.npy'), train_files)
np.save(os.path.join(meta_path, 'test_ids.npy'), test_files)

print(df['dx'].value_counts())
print(df.loc[train_files]['dx'].value_counts())
print(df.loc[test_files]['dx'].value_counts())
"""
