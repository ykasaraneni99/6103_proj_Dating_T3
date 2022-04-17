# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'


#%%
# NumPy

import numpy as np
import pandas as pd

dating = pd.read_csv("okcupid_profiles.csv")


#dating.drop(['education'], axis=1)
#dating.drop(['essay0'], axis=1)
#dating.drop(['essay1'], axis =1)

# dropping all the unequired columns.
new_data =dating.drop(['education','ethnicity','speaks','essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9','offspring','location','sign','smokes','pets','last_online','income',
'job','last_online','religion','sign','orientation'], axis=1)
print(new_data.head())
 

#%%
# checking if there are any NaN values for one column
new_data['drugs'].isnull().values.any()

# count of NaN values for that particular column
new_data['drugs'].isnull().sum()

# Droping all the column which has NAN values
new_dating_data = new_data.dropna()  

# %%

## for body type for diet

data =(new_dating_data.groupby("body_type")[["diet"]].count().sort_values(by="diet", ascending=False))
data["% of participants"]=(data["diet"]/data["diet"].sum())*100

#%%
## for body type for drinks

data_d =(new_dating_data.groupby("body_type")[["drinks"]].count().sort_values(by="drinks", ascending=False))
data_d["% of participants"]=(data_d["drinks"]/data_d["drinks"].sum())*100



#%%
import matplotlib as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.countplot(x='body_type', data=new_dating_data,
hue='sex',
order=new_dating_data['body_type'].value_counts().iloc[:10].index)
# %%
