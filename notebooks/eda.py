# %%
import pandas as pd
import matplotlib.pyplot as plt
# %%

train = pd.read_csv('../data/input/train.csv')
test = pd.read_csv('../data/input/test.csv')

# %%
print(train.isnull().sum())
print(train.shape)
print(train.info())
#%%
print(test.isnull().sum())
print(test.shape)
print(test.info())