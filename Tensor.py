#%%
import pandas as pd
import seaborn as sns
import numpy as np #numpy is for linear algebra
import os #allows for portable operating system dependant functionality
import tensorflow as tf #neural network 
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tqdm import tqdm #smart progress meter
import matplotlib.pyplot as plt

trainDir = "heart.csv"
df = pd.read_csv("heart.csv")
df.head()
df.shape #918 elements, 12 classes
df.isnull().sum()
df.info()
df.columns
#create categories for model
categories = df.select_dtypes(include=['object']).columns
categories = list(categories)
categories.append("HeartDisease")#add HeartDisease b/c we need to know if they had it
#total of 6 categories

#create initial graphs
row = 2
col = 3
column_names = categories

f, axes = plt.subplots(row, col, figsize=(15,10))
itr = 0

for r in range(row):
  for c in range(col):
    sns.countplot(data=df, x=column_names[itr], ax=axes[r,c])
    itr +=1
#end creating initial graphs
#%%


#%%