# How likely an individual is to be seeing someone else based on sex, sexual orientation, and age?
#%%
import pandas as pd 
import numpy as np
dating = pd.read_csv("/Users/urnishabhuiyan/Documents/6103_proj_Dating_T3/OkCupid_Data/okcupid_profiles.csv")
okcupid = dating.filter(["age", "status", "sex", "orientation"], axis=1)
# %%
#preprocessing the data 
def new_orientation(row):
    orientation = row["orientation"]
    if orientation == "straight": return "Straight"
    if orientation == "bisexual": return "Bisexual"
    if orientation == "gay": return "Gay"
    if orientation == " easy on the eyes.  i'm looking for friends": return np.nan 
    return orientation 
okcupid["orientation"] = okcupid.apply(new_orientation, axis=1)
print(okcupid.orientation.value_counts())

def new_status(row):
    status = row["status"]
    if status == "single": return "Yes"
    if status == "seeing someone": return "No"
    if status == "available": return "Yes"
    if status == "married": return "No"
    if status == "unknown": return "Yes"
    if status == "simple and potent people met in everyday life ... (which especially includes infants and children) athlete": return np.nan
    return status 
okcupid["status"] = okcupid.apply(new_status, axis=1)
okcupid = okcupid.rename(columns={'status': 'single'})
print(okcupid.single.value_counts())

def new_sex(row):
    sex = row["sex"]
    if sex == "m": return "Male"
    if sex == "f": return "Female"
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
# %%
#Attempting different data models to predict outcome 
import statsmodels.api as sm 
from statsmodels.formula.api import glm

# modelStatusLogitFit = glm(formula='single ~ age+C(orientation)+C(sex)+C(sex):C(orientation)+age:C(sex)+age:C(orientation)', data=okcupid, family=sm.families.Binomial()).fit()
# print( modelStatusLogitFit.summary() )

modelStatusLogitFit = glm(formula='single ~ age+C(orientation)+C(sex)+C(sex):C(orientation)+age:C(sex)', data=okcupid, family=sm.families.Binomial()).fit()
print( modelStatusLogitFit.summary() )

# modelStatusLogitFit = glm(formula='single ~ age+C(orientation)+C(sex)+age:C(sex)+age:C(orientation)', data=okcupid, family=sm.families.Binomial()).fit()
# print( modelStatusLogitFit.summary() )

newdata = {"age":[26,30,52,69] , "orientation":["Gay","Straight", "Bisexual", "Straight"], "sex":["Female","Male", "Male","Female"]}
print("Probability of Being Taken", modelStatusLogitFit.predict(newdata))
# %%
