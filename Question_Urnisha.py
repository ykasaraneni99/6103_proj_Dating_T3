# How likely an individual is to be seeing someone else based on sex, sexual orientation, and age?
#%%
import pandas as pd 
import numpy as np
import seaborn as sns
import os 
dating = pd.read_csv("/Users/urnishabhuiyan/Documents/6103_proj_Dating_T3/OkCupid_Data/okcupid_profiles.csv")
okcupid1 = dating.filter(["age", "status", "sex", "orientation"], axis=1)
okcupid = okcupid1.dropna(how="all")
print("Ready to go")
# %% 
#preprocessing the data 
# def new_orientation(row):
#     orientation = row["orientation"]
#     if orientation == "straight": return "Straight"
#     if orientation == "bisexual": return "Bisexual"
#     if orientation == "gay": return "Gay"
#     if orientation == " easy on the eyes.  i'm looking for friends": return np.nan 
#     return orientation 
# okcupid["orientation"] = okcupid.apply(new_orientation, axis=1)
# print(okcupid.orientation.value_counts())

def new_orientation(row):
    orientation = row["orientation"]
    if orientation == "straight": return 0
    if orientation == "bisexual": return 1
    if orientation == "gay": return 2
    if orientation == " easy on the eyes.  i'm looking for friends": return np.nan 
    return orientation 
okcupid["orientation"] = okcupid.apply(new_orientation, axis=1)
print(okcupid.orientation.value_counts())

# def new_status(row):
#     status = row["status"]
#     if status == "single": return "No"
#     if status == "seeing someone": return "Yes"
#     if status == "available": return "No"
#     if status == "married": return "Yes"
#     if status == "unknown": return "No"
#     if status == "simple and potent people met in everyday life ... (which especially includes infants and children) athlete": return np.nan
#     return status 
# okcupid["status"] = okcupid.apply(new_status, axis=1)
# okcupid = okcupid.rename(columns={'status': 'taken'})
# print(okcupid.taken.value_counts())

def new_status(row):
    status = row["status"]
    if status == "single": return 0
    if status == "seeing someone": return 1
    if status == "available": return 0
    if status == "married": return 1
    if status == "unknown": return 0
    if status == "simple and potent people met in everyday life ... (which especially includes infants and children) athlete": return np.nan
    return status 
okcupid["status"] = okcupid.apply(new_status, axis=1)
print(okcupid.status.value_counts())

# def new_sex(row):
#     sex = row["sex"]
#     if sex == "m": return "Male"
#     if sex == "f": return "Female"
#     if sex == " strength trainer/power lifter with brains and heart": return np.nan
#     return sex 
# okcupid["sex"] = okcupid.apply(new_sex, axis=1)
# print(okcupid.sex.value_counts())

def new_sex(row):
    sex = row["sex"]
    if sex == "m": return 0
    if sex == "f": return 1
    if sex == " strength trainer/power lifter with brains and heart": return np.nan
    return sex 
okcupid["sex"] = okcupid.apply(new_sex, axis=1)
print(okcupid.sex.value_counts())

def new_age(row):
    age = row["age"]
    try: age = int(age) 
    except: pass
    try: 
        if not isinstance(age,int) : age = float(age)  
    except: pass
  
    if ( isinstance(age,int) or isinstance(age,float) ) and not isinstance(age, bool): return ( age if age>=0 else np.nan )
    if isinstance(age, bool): return np.nan
    
    age = age.strip()
    if age == "st teachers; the powerfully humble": return np.nan
    return age 
okcupid["age"] = okcupid.apply(new_age, axis=1)
print(okcupid.age.value_counts())

okcupid_clean = okcupid.dropna(how="all")
# %%
#Attempting different linear regression interactions to predict outcome 
import statsmodels.api as sm 
from statsmodels.formula.api import glm

modelStatusLogitFit = glm(formula='status ~ age+C(orientation)+C(sex)', data=okcupid_clean, family=sm.families.Binomial()).fit()
print( modelStatusLogitFit.summary() )
#Best to use each x value seperately as the interactions gave varying insignificant p-values. 

# modelStatusLogitFit = glm(formula='taken ~ age+C(orientation)+C(sex)+C(sex):C(orientation)+age:C(sex)+age:C(orientation)', data=okcupid, family=sm.families.Binomial()).fit()
# print( modelStatusLogitFit.summary() )

# modelStatusLogit = glm(formula='taken ~ age+C(orientation)+C(sex)+C(sex):C(orientation)+age:C(sex)', data=okcupid, family=sm.families.Binomial())
# modelStatusLogitFit = modelStatusLogit.fit()
# print( modelStatusLogitFit.summary() )

# modelStatusLogitFit = glm(formula='taken ~ age+C(orientation)+C(sex)+age:C(sex)+age:C(orientation)', data=okcupid, family=sm.families.Binomial()).fit()
# print( modelStatusLogitFit.summary() )

newdata = {"age":[26] , "orientation":[0], "sex":[1]}
print("Probability of Being Taken", modelStatusLogitFit.predict(newdata))

# %%
#Determining linear regression model accuracy and precision 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

yOkcupid = okcupid_clean[['status']]
xOkcupid = okcupid_clean[['sex', 'age', 'orientation']]
x_train1, x_test1, y_train1, y_test1 = train_test_split(xOkcupid, yOkcupid, test_size=.66, random_state=1)

data_logit1 = LogisticRegression()
data_logit1.fit(x_train1, y_train1)
print('Logit Model Train Score:', data_logit1.score(x_train1, y_train1)) 
print('Logit Model Test Score', data_logit1.score(x_test1, y_test1))

predictions = data_logit1.predict(x_test1)
cm = metrics.confusion_matrix(y_test1, predictions)
print(cm)

print(confusion_matrix(y_test1, data_logit1.predict(x_test1)))
print(classification_report(y_test1, data_logit1.predict(x_test1)))
#Train and test accuracy was quite good; however, the precision and recall were not good. 
#%%
#Attempting an sklearn LinearSVC model for same data 
from sklearn.svm import LinearSVC
linearsvc = LinearSVC()
linearsvc.fit(x_train1,y_train1)
print(f'LinearSVC train score: {linearsvc.score(x_train1,y_train1)}')
print(f'LinearSVC test score:  {linearsvc.score(x_test1,y_test1)}')
print(confusion_matrix(y_test1, linearsvc.predict(x_test1)))
print(classification_report(y_test1, linearsvc.predict(x_test1)))
#The linearSVC model appears to have the same accuracy, precision, and recall results. 
#%%
#Determining the 5-number summary of data 
from numpy import percentile
from numpy.random import rand
predictedvalues = okcupid.status
predictedvalues = rand(1000)
data_min, data_max = predictedvalues.min(), predictedvalues.max()
quartiles = percentile(predictedvalues, [25,50,75])
print('Min: %.3f' % data_min)
print('Q1: %.3f' % quartiles[0])
print('Median: %.3f' % quartiles[1])
print('Q3: %.3f' % quartiles[2])
print('Max: %.3f' % data_max)
#%%
#Changing threshold from 0.5 to 0.05 
y_pred = (data_logit1.predict_proba(x_test1)[:,1] >= 0.05).astype(bool)
print(confusion_matrix(y_test1, y_pred))
print(classification_report(y_test1, y_pred))

#Changing threshold from 0.5 to 0.10
y_pred = (data_logit1.predict_proba(x_test1)[:,1] >= 0.10).astype(bool)
print(confusion_matrix(y_test1, y_pred))
print(classification_report(y_test1, y_pred))

#Changing threshold from 0.5 to 0.15
y_pred = (data_logit1.predict_proba(x_test1)[:,1] >= 0.15).astype(bool)
print(confusion_matrix(y_test1, y_pred))
print(classification_report(y_test1, y_pred))
#Appears the best threshold to use for this dataset is 0.10. This gave a precision of 0.12 and recall of 0.03. It's not very good still; however, it is a bit better from the original numbers. This brings down the accuracy by 1 percent, but allows for a bit more precision and recall. 
# %%
#Visualizing the unbalanced data
def new_status(row):
    status = row["status"]
    if status == "single": return "No"
    if status == "seeing someone": return "Yes"
    if status == "available": return "No"
    if status == "married": return "Yes"
    if status == "unknown": return "No"
    if status == "simple and potent people met in everyday life ... (which especially includes infants and children) athlete": return np.nan
    return status 
okcupid["status"] = okcupid.apply(new_status, axis=1)

def new_sex(row):
    sex = row["sex"]
    if sex == "m": return "Male"
    if sex == "f": return "Female"
    if sex == " strength trainer/power lifter with brains and heart": return np.nan
    return sex 
okcupid["sex"] = okcupid.apply(new_sex, axis=1)


plt.figure(figsize=(10,8))
sns.countplot(x="status", hue="sex", data=okcupid).set(title="Count of Individuals Listed as In a Relationship per Sex", xlabel="In a Relationship", ylabel="Count")
plt.legend(title="Sex")
plt.show()
# %%
