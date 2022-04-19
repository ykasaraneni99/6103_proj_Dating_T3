# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'


#%%
# NumPy

import numpy as np
import pandas as pd

dating = pd.read_csv("okcupid_profiles.csv")


# dropping all the unrequired columns.
new_data =dating.drop(['education','ethnicity','speaks','essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9','offspring','location','sign','pets','last_online','income',
'job','last_online','religion','sign','orientation'], axis=1)
print(new_data.head())

#%%
new_data.isna().sum()
#%%
new_data_a = new_data.dropna(thresh = 13, how = 'any')
new_data_a.isna().sum()


new_data.describe()

new_data.dtypes
 

#%%


# Droping all the column which has NAN values
new_dating_data = new_data.dropna()  

#%%
replace_ty={'body_type':{"a little extra":1,"average":2,"athletic":3,"skinny":4,"thin":5, "fit":6, "curvy":7, "full":9,"full figured":10, "jacked":11, "overweight":12, "used up":13, "rather not say":14},
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
from matplotlib.pyplot import figure, legend
import matplotlib.pyplot as plt


#%%
plt.figure(figsize=(10, 7))
sns.countplot(x='body_type', data=df_dating,
hue='diet', 
order=df_dating['body_type'].value_counts().iloc[:10].index).set(title = 'body type count per diet',xlabel='body type', ylabel = 'count')
#plt.legend(labels=['strictly anything','mostly other', 'mostly anything','mostly vegetarian','strictly vegan', 'anything', 'vegetarian', 'mostly halal', 'strictly vegetarian', 'other', 'strictly other', 'vegan', 'mostly vegan', 'mostly kosher', 'strictly halal', 'halal', 'strictly kosher', 'kosher'])

plt.figure(figsize=(10, 5))
sns.countplot(x='body_type', data=df_dating,
hue='sex',
order=df_dating['body_type'].value_counts().iloc[:10].index)
#%%

plt.figure(figsize=(10, 6))
sns.countplot(x='body_type', data=df_dating,
hue='drinks',
order=df_dating['body_type'].value_counts().iloc[:10].index).set(title = 'body type count per drink',xlabel='body type', ylabel = 'count')
plt.legend(labels=['socially',"often","not at all","rarely", "very often", 'desperately'])



# %%

plt.figure(figsize=(10, 5))
sns.countplot(x='body_type', data=df_dating,
hue='drugs', palette='Oranges',
order=df_dating['body_type'].value_counts().iloc[:10].index).set(title = 'body type count per drugs',xlabel='body type', ylabel = 'count')
plt.legend(labels=['never','sometimes','often'])


#%%
plt.figure(figsize=(10, 6))
sns.countplot(x='body_type', data=df_dating,
hue='smokes',
order=df_dating['body_type'].value_counts().iloc[:10].index).set(title = 'body type count per smoke',xlabel='body type', ylabel = 'count')
plt.legend(labels=['sometimes', 'never', 'trying to quit', 'when drinking', 'yes'])


#%%
from statsmodels.formula.api import glm
import statsmodels.api as sm

modelTestLogit = glm(formula='body_type ~ C(diet) + C(drinks) + C(smokes) + C(drugs)', data=df_dating)
modelTestLogitFit = modelTestLogit.fit()
print( modelTestLogitFit.summary())


# %%
# Let's try logistic regression with sklearn 

# Prepare our X data (features, predictors, regressors) and y data (target, dependent variable)
xdata = df_dating[['diet','drinks','smokes','drugs']]
ydata = df_dating['body_type']
print(type(xdata))
print(type(ydata))



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, random_state=1 , test_size=0.75)

print('x_train type',type(x_train))
print('x_trainshape',x_train.shape)
print('x_test type',type(x_test))
print('x_test shape',x_test.shape)
print('y_train type',type(y_train))
print('y_train shape',y_train.shape)
print('y_test type',type(y_test))
print('y_test shape',y_test.shape)

print("\nReady to continue.")


#%%
from sklearn.linear_model import LogisticRegression

logitr = LogisticRegression()  
logitr.fit(x_train, y_train)
print('Logit model accuracy (with the test set):', logitr.score(x_test, y_test))
print('Logit model accuracy (with the train set):', logitr.score(x_train, y_train))

print("\nReady to continue.")

#%%
print(logitr.predict(x_test))

print("\nReady to continue.")

#%%
print(logitr.predict_proba(x_train[:5]))
print(logitr.predict_proba(x_test[:5]))

print("\nReady to continue.")

#%%
test = logitr.predict_proba(x_test)
type(test)

print("\nReady to continue.")

# %%
cut_off = 0.7
predictions = (logitr.predict_proba(x_test)[:,1]>cut_off).astype(int)
print(predictions)

print("\nReady to continue.")

#%%
def predictcutoff(arr, cutoff):
  arrbool = arr[:,1]>cutoff
  arr= arr[:,1]*arrbool/arr[:,1]
  # arr= arr[:,1]*arrbool
  return arr.astype(int)

test = logitr.predict_proba(x_test)
p = predictcutoff(test, 0.9)
print(p)

# print("\nReady to continue.")

#%%

predictcutoff(test, 0.5)

# classification_report

from sklearn.metrics import classification_report
y_true, y_pred = y_test, logitr.predict(x_test)
print(classification_report(y_true, y_pred))

# %%
x1data = df_dating[['drugs']]
y1data = df_dating['body_type']
print(type(x1data))
print(type(y1data))


# %%
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1data, y1data, random_state=1 , test_size=0.50)

print('x_train type',type(x1_train))
print('x_trainshape',x1_train.shape)
print('x_test type',type(x1_test))
print('x_test shape',x1_test.shape)
print('y_train type',type(y1_train))
print('y_train shape',y1_train.shape)
print('y_test type',type(y1_test))
print('y_test shape',y1_test.shape)

print("\nReady to continue.")


#%%
from sklearn.linear_model import LogisticRegression

logitr_d = LogisticRegression()  
logitr_d.fit(x1_train, y1_train)
print('Logit model accuracy (with the test set):', logitr_d.score(x1_test, y1_test))
print('Logit model accuracy (with the train set):', logitr_d.score(x1_train, y1_train))

print("\nReady to continue.")

#%%
print(logitr_d.predict(x1_test))

print("\nReady to continue.")

#%%
print(logitr_d.predict_proba(x1_train[:5]))
print(logitr_d.predict_proba(x1_test[:5]))

print("\nReady to continue.")

#%%
test = logitr_d.predict_proba(x1_test)
type(test)

print("\nReady to continue.")

# %%
cut_off = 0.7
predictions = (logitr_d.predict_proba(x1_test)[:,1]>cut_off).astype(int)
print(predictions)

print("\nReady to continue.")

#%%
def predictcutoff(arr, cutoff):
  arrbool = arr[:,1]>cutoff
  arr= arr[:,1]*arrbool/arr[:,1]
  # arr= arr[:,1]*arrbool
  return arr.astype(int)

test1 = logitr_d.predict_proba(x1_test)
p = predictcutoff(test1, 0.9)
print(p)

# print("\nReady to continue.")

#%%

predictcutoff(test1, 0.5)

# %%
