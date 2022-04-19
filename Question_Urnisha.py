# How likely an individual is to be seeing someone else based on sex, sexual orientation, and age?
#%%
import pandas as pd 
import numpy as np
import seaborn as sns
dating = pd.read_csv("/Users/urnishabhuiyan/Documents/6103_proj_Dating_T3/OkCupid_Data/okcupid_profiles.csv")
okcupid1 = dating.filter(["age", "status", "sex", "orientation"], axis=1)
okcupid = okcupid1.dropna(how="all")
# %%
# preprocessing the data 
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
#Attempting different data models to predict outcome 
import statsmodels.api as sm 
from statsmodels.formula.api import glm

modelStatusLogitFit = glm(formula='status ~ age+C(orientation)+C(sex)', data=okcupid_clean, family=sm.families.Binomial()).fit()
print( modelStatusLogitFit.summary() )

# modelStatusLogitFit = glm(formula='taken ~ age+C(orientation)+C(sex)+C(sex):C(orientation)+age:C(sex)+age:C(orientation)', data=okcupid, family=sm.families.Binomial()).fit()
# print( modelStatusLogitFit.summary() )

# modelStatusLogit = glm(formula='taken ~ age+C(orientation)+C(sex)+C(sex):C(orientation)+age:C(sex)', data=okcupid, family=sm.families.Binomial())
# modelStatusLogitFit = modelStatusLogit.fit()
# print( modelStatusLogitFit.summary() )

# modelStatusLogitFit = glm(formula='taken ~ age+C(orientation)+C(sex)+age:C(sex)+age:C(orientation)', data=okcupid, family=sm.families.Binomial()).fit()
# print( modelStatusLogitFit.summary() )

newdata = {"age":[26] , "orientation":[0], "sex":[1]}
print("Probability of Being Taken", modelStatusLogitFit.predict(newdata))
#%%
okcupidpredicitons = pd.DataFrame( columns=['total_status'], data=modelStatusLogitFit.predict(okcupid_clean))
cut_off = [0.3,0.5,0.7]
for x in cut_off:
    okcupidpredicitons['statusALLlogit'] = np.where(okcupidpredicitons["total_status"] > x, 1, 0)
    okcupid_cmatrix = pd.crosstab(okcupid.status, okcupidpredicitons.statusALLlogit,
    rownames=['Actual'], colnames=['Predicted'],
    margins = True)
#     print(f"Total Accuracy of the Model with {x} Cutoff: {(okcupid_cmatrix.iloc[1,1] + okcupid_cmatrix.iloc[0,0])/ okcupid_cmatrix.iloc[2,2]} ")
#     print(f"Precision of the Model with {x} Cutoff: { okcupid_cmatrix.iloc[1,1] / (okcupid_cmatrix.iloc[1,1] + okcupid_cmatrix.iloc[1,0] ) } ")
#     print(f"Recall Rate of the Model with {x} Cutoff: { (okcupid_cmatrix.iloc[1,1] / (okcupid_cmatrix.iloc[1,1] + okcupid_cmatrix.iloc[0,1]))} ")
# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

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

precision = metrics.precision_score(y_test1, predictions)
accuracy = metrics.accuracy_score(y_test1, predictions)
print("precision metrics:", precision)
print("accuracy score:", accuracy)
# %%
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
