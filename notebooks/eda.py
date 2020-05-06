# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# %%

train = pd.read_csv('../data/input/train.csv')
test = pd.read_csv('../data/input/test.csv')

# %%
print(train.isnull().sum())
print(train.shape)
print(train.info())
# %%
print(test.isnull().sum())
print(test.shape)
print(test.info())

# %%
columns = test.columns
print(columns)
for col in columns:
    print(col)
    print(len(train[col].unique()))
    print('_'*30)

# %%

train['title'].apply(lambda x: 'official' in x.lower()).sum()

# %%
train['title'].apply(lambda x: '公式' in x.lower()).sum()
