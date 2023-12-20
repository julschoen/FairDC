import pandas as pd
import numpy as np
import os
from skimage import io

meta_path = '../../Downloads/'
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
