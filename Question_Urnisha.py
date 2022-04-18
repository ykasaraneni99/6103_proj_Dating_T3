# How likely an individual is to be seeing someone else based on sex, sexual orientation, and height?
#%%
import pandas as pd 
import numpy as np
dating = pd.read_csv("/Users/urnishabhuiyan/Documents/6103_proj_Dating_T3/OkCupid_Data/okcupid_profiles.csv")
okcupid = dating.filter(["age", "status", "sex", "orientation", "body_type", "diet", "drinks", "drugs"], axis=1)
# %%
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
    if status == "single": return "Single"
    if status == "seeing someone": return "Seeing Someone"
    if status == "available": return "Available"
    if status == "married": return "Married"
    if status == "unknown": return np.nan
    if status == "simple and potent people met in everyday life ... (which especially includes infants and children) athlete": return np.nan
    return status 
okcupid["status"] = okcupid.apply(new_status, axis=1)
print(okcupid.status.value_counts())

def new_sex(row):
    sex = row["sex"]
    if sex == "m": return "Male"
    if sex == "f": return "Female"
    if sex == " strength trainer/power lifter with brains and heart": return np.nan
    return sex 
okcupid["sex"] = okcupid.apply(new_sex, axis=1)
print(okcupid.sex.value_counts())
# %%
