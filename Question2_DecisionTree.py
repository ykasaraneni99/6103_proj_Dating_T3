#Q2. How does diet, drinking, drugs, and smoking impact one's body type?
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
#%%
dating = pd.read_csv("/Users/yasaswikasaraneni/Documents/GW/6103_proj_Dating_T3/OkCupid_Data/okcupid_profiles.csv")
okcupid = dating.filter(["body_type", "diet", "drinks", "drugs", "smokes","status","age","sex"], axis=1)

# %%
new_okcupid = okcupid.dropna() 


# %%
#Preprocessing
replace_ty = {'body_type':{"a little extra":0,"average":1,"athletic":2,"skinny":3,"thin":4, 'fit':5, 'curvy':6, 'full figured':7, 'jacked':8, 'overweight':9, 'used up':10, 'rather not say':11, ' community':12},
            'diet':{"vegetarian":1, "mostly vegetarian":2, "strictly vegetarian":3, "vegan":4, "mostly vegan":5, "strictly vegan":6, "halal":7, "mostly halal":8, "strictly halal":9, "kosher":10, "mostly kosher":11, "strictly kosher":12, "anything":13, "mostly anything":14, "strictly anything":15, "other":16, "mostly other":17, "strictly other":18, " dating partner(s)":19},
            'drinks':{"socially":1, "often":2, "not at all":3, "rarely":4, "very often":5, "desperately":6, " playmates":7},
            'drugs':{"never":1, "sometimes":2, "often":3, " and maybe down the line":4},
            'smokes':{"sometimes":1, "no":2, "trying to quit":3, "when drinking":4, "yes":5, " seems important to say":6},
            'status':{'single':1, 'available':2, 'seeing someone':3, 'married':4, 'unknown':5,'simple and potent people met in everyday life ... (which especially includes infants and children) athlete':6},
            'sex':{'m':0, 'f':2, ' strength trainer/power lifter with brains and heart':3}
             }

# %%
df_okcupid=new_okcupid.replace(replace_ty)

#%%
# Correlation Matrix
cormat = df_okcupid.corr()
round(cormat,2)

#%%
# check missing values in variables
df_okcupid.isnull().sum()

#%%
sns.countplot(data=new_okcupid, y="body_type");
#it seems that most users will describe themselves as average, fit, or athletic

#%%
sns.countplot(data=new_okcupid, y="body_type", hue = "sex");
#This chart shows the break down of body type by gender and it seems that some of the body type descriptions are highly gendered. For example "curvy" and "full figured" are highly female descriptions, while males use "a little extra", and "overweight" more often. 

#%%
sns.countplot(data=new_okcupid, y="diet");
#Here is a chart of the dietary information for users. Most user eat "mostly anything", followed by "anything", and "strictly anything", being open-minded seems to be a popular signal to potential partners.

#%%
sns.countplot(data=new_okcupid, y="drinks");
#The majority of the users drink "socially", then "rarely" and "often".

#%%
sns.countplot(data=new_okcupid, y="drugs");
#The vast majority of users "never" use drugs.

#%%
sns.countplot(data=new_okcupid, y="smokes");
#Similarly for drugs the majority of users chose "no" for smoking.

#%%
sns.countplot(data=new_okcupid, y="status");
#The relationship status for a dating website is fairly predictable. One would assume that most people are single and available which is reflected in the data.

#%%
# Histogram Plot of Body Type
plt.hist(df_okcupid.body_type)
plt.xlabel("Body Type")
plt.ylabel("Count")
plt.title("Histogram of Body Type")
plt.show()

#%%
# Histogram Plot of Diet
plt.hist(df_okcupid.diet)
plt.xlabel("Diet")
plt.ylabel("Count")
plt.title("Histogram of Diet")
plt.show()

#%%
plt.hist(df_okcupid.diet, 35, range=[0, 19], facecolor='red', align='mid')
plt.xlabel("Diet")
plt.ylabel("Count")
plt.title("Histogram of Diet")

#%%
# Histogram Plot of Drinks
plt.hist(df_okcupid.drinks)
plt.xlabel("Drinks")
plt.ylabel("Count")
plt.title("Histogram of Drinks")
plt.show()

#%%
# Histogram Plot of Drugs
plt.hist(df_okcupid.drugs)
plt.xlabel("Drugs")
plt.ylabel("Count")
plt.title("Histogram of Drugs")
plt.show()

#%%
# Histogram Plot of Smokes
plt.hist(df_okcupid.smokes)
plt.xlabel("Smokes")
plt.ylabel("Count")
plt.title("Histogram of Smokes")
plt.show()

#%%
# Histogram Plot of Status
plt.hist(df_okcupid.status)
plt.xlabel("Status")
plt.ylabel("Count")
plt.title("Histogram of Status")
plt.show()

#%%
# Scatter Plot between Body Type and Status
plt.scatter(df_okcupid.body_type, df_okcupid.status, color = "green")
plt.xlabel("Body Type")
plt.ylabel("Status")
plt.title("Body Type vs Status Scatter")
plt.show()

#%%
plt.scatter(df_okcupid.diet, df_okcupid.sex, color = "green")
plt.xlabel("Diet")
plt.ylabel("Sex")
plt.title("Diet vs Sex Scatter plot")
plt.show()

#%%
# Declare feature vector and target variable
X = df_okcupid.drop(['body_type','status','sex','age'], axis=1)
y = df_okcupid['body_type']

#%% 
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

#%%
# check the shape of X_train and X_test
X_train.shape, X_test.shape

#%%
# check data types in X_train
X_train.dtypes

#%%
## Decision Tree Classifier with criterion gini index
# import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

#%%
# instantiate the DecisionTreeClassifier model with criterion gini index
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

# fit the model
clf_gini.fit(X_train, y_train)

#%%
# Predict the Test set results with criterion gini index
y_pred_gini = clf_gini.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_gini))

#%%
# Checking accuracy score with criterion gini index
from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

#%%
# Comparing the train-set and test-set accuracy
y_pred_train_gini = clf_gini.predict(X_train)
y_pred_train_gini
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))

#%%
# Checking for overfitting and underfitting
# print the scores on training and test set
print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))

#%%
## Decision Tree Classifier with criterion entropy
# instantiate the DecisionTreeClassifier model with criterion entropy
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

# fit the model
clf_en.fit(X_train, y_train)

#%%
#Predict the Test set results with criterion entropy
y_pred_en = clf_en.predict(X_test)

#%%
#Check accuracy score with criterion entropy
from sklearn.metrics import accuracy_score
print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))

#%%
#Compare the train-set and test-set accuracy
y_pred_train_en = clf_en.predict(X_train)
y_pred_train_en

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))

#%%
#Check for overfitting and underfitting
# print the scores on training and test set
print('Training set score: {:.4f}'.format(clf_en.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(clf_en.score(X_test, y_test)))

#%%
##Confusion matrix
# Print the Confusion Matrix and slice it into four pieces
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_en)
print('Confusion matrix\n\n', cm)

#%%
# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_gini))




#%%[markdown]
#Different method - not correct (just tried)
# %%
cols = ['body_type', 'diet', 'drinks', 'drugs', 'smokes']
df = okcupid[cols].dropna()
# %%
for col in cols[:-1]:
    df = pd.get_dummies(df, columns=[col], prefix = [col])
# %%
df.head()
# %%
col_length = len(df.columns)
# %%
X = df.iloc[:, 1:col_length]
Y = df.iloc[:, 0:1]
# %%
from sklearn.model_selection import train_test_split 
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state = 0)
# %%
Y_train = Y_train.to_numpy().ravel()
Y_val = Y_val.to_numpy().ravel()
Y_train
# %%
Y_val
# %%
from sklearn.tree import DecisionTreeClassifier

cart_model = DecisionTreeClassifier().fit(X_train, Y_train) 
cart_predictions = cart_model.predict(X_train) 
# %%
from sklearn.metrics import classification_report
print(classification_report(Y_train, cart_predictions))
# %%
from sklearn.metrics import confusion_matrix 
cart_cm = confusion_matrix(Y_train, cart_predictions)
cart_labels = cart_model.classes_
cart_cm

# %%
