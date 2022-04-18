# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'


#%%
# NumPy

import numpy as np
import pandas as pd

dating = pd.read_csv("okcupid_profiles.csv")


# dropping all the unequired columns.
new_data =dating.drop(['education','ethnicity','speaks','essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9','offspring','location','sign','pets','last_online','income',
'job','last_online','religion','sign','orientation'], axis=1)
print(new_data.head())

new_data.isna().sum()

new_data_a = new_data.dropna(thresh = 13, how = 'any')
new_data_a.isna().sum()


new_data.describe()

new_data.dtypes
 

#%%

#
dating.corr().style.background_gradient()

# checking if there are any NaN values for one column
new_data['drugs'].isnull().values.any()

# count of NaN values for that particular column
new_data['drugs'].isnull().sum()

# Droping all the column which has NAN values
new_dating_data = new_data.dropna()  

#%%
replace_ty={#'body_type':{"a little extra":1,"average":2,"athletic":3,"skinny":4,"thin":5, "fit":6, "curvy":7, "full":9,"full figured":10, "jacked":11, "overweight":12, "used up":13, "rather not say":14},
            'drinks':{'socially':1,"often":2,"not at all":3,"rarely":4, "very often":5, 'desperately' :6},
            'diet':{'strictly anything':1,'mostly other':2, 'mostly anything':3,'mostly vegetarian':4,'strictly vegan':5, 'anything':6, 'vegetarian':7, 'mostly halal':8, 'strictly vegetarian':9, 'other':10, 'strictly other': 11, 'vegan':12, 'mostly vegan':13, 'mostly kosher':14, 'strictly halal':15, 'halal':16, 'strictly kosher':17, 'kosher': 18},
            'drugs':{'never': 1, 'sometimes': 2, 'often': 3},
            'smokes':{'sometimes':1, 'no':2, 'trying to quit':3, 'when drinking':4, 'yes':5}
           }

df_dating=new_dating_data.replace(replace_ty)
#%%
## body type for people who are in diet

data =(new_dating_data.groupby("body_type")[["diet"]].count().sort_values(by="diet", ascending=False))
data["% of participants"]=(data["diet"]/data["diet"].sum())*100
data


#%%
## body type for people who drink

data_d =(new_dating_data.groupby("body_type")[["drinks"]].count().sort_values(by="body_type", ascending=False))
data_d["% of participants"]=(data_d["drinks"]/data_d["drinks"].sum())*100
data_d


#%%
## body type for people who take drugs

data_drug =(new_dating_data.groupby("body_type")[["drugs"]].count().sort_values(by="body_type", ascending=True))
data_drug["% of participants"]=(data_drug["drugs"]/data_drug["drugs"].sum())*100
data_drug

#%%
data_smoke =(new_dating_data.groupby("body_type")[["smokes"]].count().sort_values(by="body_type", ascending=False))
data_smoke["% of participants"]=(data_smoke["smokes"]/data_smoke["smokes"].sum())*100
data_smoke

#%%
import matplotlib as plt
import seaborn as sns
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
sns.set_theme(style="whitegrid")
sns.boxenplot(x="body_type", y="age",data=df_dating)

#df_dating.plot('diet','body_type', kind='scatter', marker='o') 
#plt.ylabel('body type')
#plt.xlabel('diet')
#plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x='body_type', data=df_dating,
hue='drinks',
order=df_dating['body_type'].value_counts().iloc[:10].index)

#%%
plt.figure(figsize=(10, 7))
sns.countplot(x='body_type', data=df_dating,
hue='diet',
order=df_dating['body_type'].value_counts().iloc[:10].index)


plt.figure(figsize=(10, 5))
sns.countplot(x='diet', data=df_dating,
hue='sex',
order=df_dating['diet'].value_counts().iloc[:10].index)
#%%

plt.figure(figsize=(10, 6))
sns.countplot(x='body_type', data=df_dating,
hue='drinks',
order=df_dating['body_type'].value_counts().iloc[:10].index)

plt.figure(figsize=(10, 5))
sns.countplot(x='drinks', data=df_dating,
hue='sex', palette='flare',
order=df_dating['drinks'].value_counts().iloc[:10].index)

# %%

plt.figure(figsize=(10, 5))
sns.countplot(x='body_type', data=df_dating,
hue='drugs',
order=df_dating['body_type'].value_counts().iloc[:10].index)

plt.figure(figsize=(10, 5))
sns.countplot(y='drugs', data=df_dating,
hue='sex', palette='Reds',
order=df_dating['drugs'].value_counts().iloc[:10].index)

#%%
plt.figure(figsize=(10, 6))
sns.countplot(x='body_type', data=df_dating,
hue='smokes',
order=df_dating['body_type'].value_counts().iloc[:10].index)

plt.figure(figsize=(10, 5))
sns.countplot(y='smokes', data=df_dating,
hue='sex', palette='Oranges',
order=df_dating['smokes'].value_counts().iloc[:10].index)
# %%
# Let's try logistic regression again with sklearn 

# Prepare our X data (features, predictors, regressors) and y data (target, dependent variable)
xdata = df_dating[['diet','drinks','smokes','drugs']]
ydata = df_dating['body_type']
print(type(xdata))
print(type(ydata))
# These xdfadmit and ydfadmit are dataframes


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, random_state=1 )

print('x_train type',type(x_train))
print('x_trainshape',x_train.shape)
print('x_test type',type(x_test))
print('x_test shape',x_test.shape)
print('y_train type',type(y_train))
print('y_train shape',y_train.shape)
print('y_test type',type(y_test))
print('y_test shape',y_test.shape)

print("\nReady to continue.")

# %%
