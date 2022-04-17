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
new_data =dating.drop(['education','ethnicity','speaks','essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9','offspring','location','sign','pets','last_online','income',
'job','last_online','religion','sign','orientation'], axis=1)
print(new_data.head())
 

#%%
# checking if there are any NaN values for one column
new_data['drugs'].isnull().values.any()

# count of NaN values for that particular column
new_data['drugs'].isnull().sum()

# Droping all the column which has NAN values
new_dating_data = new_data.dropna()  

#%%
replace_ty={'body_type':{"a little extra":1,"average":2,"athletic":3,"skinny":4,"thin":5, "fit":6, "curvy":7, "full":9,"full figured":10, "jacked":11, "overweight":12, "used up":13, "rather not say":14},
            'drinks':{'socially':15,"often":16,"not at all":17,"rarely":18, "very often":19},
           }

df_dating=new_dating_data.replace(replace_ty)
#%%
## body type for people who are in diet

data =(new_dating_data.groupby("body_type")[["diet"]].count().sort_values(by="diet", ascending=False))
data["% of participants"]=(data["diet"]/data["diet"].sum())*100
data

#%%
## body type for people who drink

data_d =(new_dating_data.groupby("body_type")[["drinks"]].count().sort_values(by="drinks", ascending=False))
data_d["% of participants"]=(data_d["drinks"]/data_d["drinks"].sum())*100
data_d

#%%
## body type for people who take drugs

data_drug =(new_dating_data.groupby("body_type")[["drugs"]].count().sort_values(by="drugs", ascending=False))
data_drug["% of participants"]=(data_drug["drugs"]/data_drug["drugs"].sum())*100
data_drug

#%%
data_smoke =(new_dating_data.groupby("body_type")[["smokes"]].count().sort_values(by="smokes", ascending=False))
data_smoke["% of participants"]=(data_smoke["smokes"]/data_smoke["smokes"].sum())*100
data_smoke

#%%
import matplotlib as plt
import seaborn as sns
from matplotlib.pyplot import figure

plt.figure(figsize=(10, 5))
sns.countplot(x='diet', data=df_dating,
hue='sex',
order=df_dating['diet'].value_counts().iloc[:10].index)
# %%

plt.figure(figsize=(15, 5))
sns.countplot(x='drinks', data=df_dating,
hue='sex',
order=df_dating['drinks'].value_counts().iloc[:10].index)
# %%

plt.figure(figsize=(15, 5))
sns.countplot(x='drugs', data=new_dating_data,
hue='sex',
order=new_dating_data['drugs'].value_counts().iloc[:10].index)
# %%
