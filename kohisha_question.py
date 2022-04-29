# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'


#%%
# NumPy

import numpy as np
import pandas as pd
from regex import B

# loading the data set
dating = pd.read_csv("okcupid_profiles.csv")

#%%
# dropping all the unrequired columns which are not related to our problem statement.
new_data =dating.drop(['education','ethnicity','speaks','essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9','offspring','location','sign','pets','last_online','income',
'job','last_online','religion','sign','orientation'], axis=1)
print(new_data.head())

#%%

#checking if there are any misssing values and printing the sum
new_data.isna().sum()

# viewing some basic statistical details like percentile, mean, std etc.
new_data.describe()

# Checking all varaiables datatypes
new_data.dtypes
 

#%%
# Droping all the column which has NAN values
new_dating_data = new_data.dropna()  

#%%
# replacing categorical values to numerical values for better modeling as there are multiple types in each variable.
replace_ty={'body_type':{"a little extra":1,"average":2,"athletic":3,"skinny":4,"thin":5, "fit":6, "curvy":7, "full":9,"full figured":10, "jacked":11, "overweight":12, "used up":13, "rather not say":14},
            'drinks':{'socially':1,"often":2,"not at all":3,"rarely":4, "very often":5, 'desperately' :6},
            'diet':{'strictly anything':1,'mostly other':2, 'mostly anything':3,'mostly vegetarian':4,'strictly vegan':5, 'anything':6, 'vegetarian':7, 'mostly halal':8, 'strictly vegetarian':9, 'other':10, 'strictly other': 11, 'vegan':12, 'mostly vegan':13, 'mostly kosher':14, 'strictly halal':15, 'halal':16, 'strictly kosher':17, 'kosher': 18},
            'drugs':{'never': 1, 'sometimes': 2, 'often': 3},
            'smokes':{'sometimes':1, 'no':2, 'trying to quit':3, 'when drinking':4, 'yes':5}
           }

df_dating=new_dating_data.replace(replace_ty)
#%%
## body type for people who are in drugs

data =(new_dating_data.groupby("body_type")[["drugs"]].count().sort_values(by="drugs", ascending=False))
data["% of participants"]=(data["drugs"]/data["drugs"].sum())*100
data


#%%
import matplotlib as plt
import seaborn as sns
from matplotlib.pyplot import figure, legend
import matplotlib.pyplot as plt


#%%
# doing preliminary analysis of data through graphs.

plt.figure(figsize=(10, 7))
sns.countplot(x='body_type', data=df_dating,
hue='diet', 
order=df_dating['body_type'].value_counts().iloc[:10].index).set(title = 'body type count per diet',xlabel='body type', ylabel = 'count')
plt.legend(loc='right', labels=['strictly anything','mostly other', 'mostly anything','mostly vegetarian','strictly vegan', 'anything', 'vegetarian', 'mostly halal', 'strictly vegetarian', 'other', 'strictly other', 'vegan', 'mostly vegan', 'mostly kosher', 'strictly halal', 'halal', 'strictly kosher', 'kosher'])

# 
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


# the p-values are high for all independent varaibles with respect to dependent varaiable which shows that there is no relation between them.
# but for drugs the p-value is low and it appears that drug can impact with body type.


# %%
# Let's try logistic regression with sklearn 

# Prepare our X data (features, predictors, regressors) and y data (target, dependent variable)
# testing the above model
xdata = df_dating[['diet','drinks','smokes','drugs']]
ydata = df_dating['body_type']
print(type(xdata))
print(type(ydata))


# model evaluation using train and test sets 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, random_state=1 , test_size=0.75)
full_split1 = linear_model.LinearRegression() # new instance
full_split1.fit(x_train, y_train)
y_pred1 = full_split1.predict(x_test)
full_split1.score(x_test, y_test)

print('score (train):', full_split1.score(x_train, y_train)) 
print('score (test):', full_split1.score(x_test, y_test)) 
print('intercept:', full_split1.intercept_) 
print('coef_:', full_split1.coef_)

#%%
x_train1, x_test1, y_train1, y_test1 = train_test_split(xdata, ydata, random_state=1234 , test_size=0.05)
full_split1 = linear_model.LinearRegression() # new instance
full_split1.fit(x_train1, y_train1)
y_pred1 = full_split1.predict(x_test1)
full_split1.score(x_test1, y_test1)

print('score (train):', full_split1.score(x_train1, y_train1)) 
print('score (test):', full_split1.score(x_test1, y_test1)) 
print('intercept:', full_split1.intercept_) 
print('coef_:', full_split1.coef_)



#%%

# checking the accuracy for the model

logitr = LogisticRegression()  
logitr.fit(x_train, y_train)
logitr.fit(x_train1, y_train1)
print('Logit model accuracy (with the test set):', logitr.score(x_test, y_test))
print('Logit model accuracy (with the train set):', logitr.score(x_train, y_train))
print('Logit model accuracy (with the train set):', logitr.score(x_test1, y_test1))
print('Logit model accuracy (with the train set):', logitr.score(x_train1, y_train1))


print("\nReady to continue.")

# the accuracy for test and train datasets are similar despite low accuracy of the model.

#%%
print(logitr.predict_proba(x_train[:2]))
print(logitr.predict_proba(x_test[:2]))

print("\nReady to continue.")
# %%


cut_off = 0.7
predictions = (logitr.predict_proba(x_test)[:,1]>cut_off).astype(int)
print(predictions)

#%%
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
full_split1 = linear_model.LinearRegression() # new instance
full_split1.fit(x1_train, y1_train)
y_pred1 = full_split1.predict(x1_test)
full_split1.score(x1_test, y1_test)

print('score (train):', full_split1.score(x1_train, y1_train)) # 0023100973484375675
print('score (test):', full_split1.score(x1_test, y1_test)) # 0.001095211027081322
print('intercept:', full_split1.intercept_) # 3.848879185046637
print('coef_:', full_split1.coef_)

#%%
x2_train1, x2_test1, y2_train1, y2_test1 = train_test_split(x1data, y1data, random_state=1234 , test_size=0.90)
full_split1 = linear_model.LinearRegression() # new instance
full_split1.fit(x2_train1, y2_train1)
y_pred1 = full_split1.predict(x2_test1)
full_split1.score(x2_test1, y2_test1)

print('score (train):', full_split1.score(x2_train1, y2_train1)) # 0.0032198745167399956
print('score (test):', full_split1.score(x2_test1, y2_test1)) # -0.00034101283397536264
print('intercept:', full_split1.intercept_) # 3.8798764612617305
print('coef_:', full_split1.coef_)


#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

logitr_d = LogisticRegression()  
logitr_d.fit(x1_train, y1_train)
logitr_d.fit(x2_train1, y2_train1)
print(f'Logit model accuracy (with the test set): {logitr_d.score(x1_test, y1_test)}')
print('Logit model accuracy (with the train set):', logitr_d.score(x1_train, y1_train))

print(f'Logit model accuracy (with the test set): {logitr_d.score(x2_test1, y2_test1)}')
print('Logit model accuracy (with the train set):', logitr_d.score(x2_train1, y2_train1))
print(confusion_matrix(y1_test, logitr_d.predict(x1_test)))
print(classification_report(y1_test, logitr_d.predict(x1_test)))

# even though the p-values for body type and drugs is significant the accuracy of the model is very low.


#%%
print(logitr_d.predict_proba(x1_train[:2]))
print(logitr_d.predict_proba(x1_test[:2]))

print("\nReady to continue.")


# %%
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

cut_off = 0.7
predictions = (logitr_d.predict_proba(x1_test)[:,1]>cut_off).astype(int)
print(predictions)
print(confusion_matrix(y1_test, y1_train))
print(classification_report(y1_test, y1_train))

# %%

# we can conclude that we cannot relate body type with smoke, drug, diet, drink accurately 
# this arises from the facts that the data provided to an dating website maynot be 100% true and there are lots of missing values in the data.