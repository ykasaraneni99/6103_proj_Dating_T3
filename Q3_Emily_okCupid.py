#%% Trying question 4: 
# Explore a relationship between user defined body type and listed hobbies with Feature Selection by chi2 analysis and KNN
## Essay Information

# essay0- My self summary
# essay1- What I’m doing with my life
# essay2- I’m really good at
# essay3- The first thing people usually notice about me
# essay4- Favorite books, movies, show, music, and food
# essay5- The six things I could never do without
# essay6- I spend a lot of time thinking about
# essay7- On a typical Friday night I am
# essay8- The most private thing I am willing to admit
# essay9- You should message me if...

from multiprocessing.reduction import duplicate
from black import get_features_used
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 
from sklearn.cluster import KMeans 
import numpy as np
from sympy import rotations

#%%

# Standard data quick checks
def dfChk(dframe, valCnt = True): 
  cnt = 1
  print('\ndataframe Basic Check function -')
  
  try:
    print(f'\n{cnt}: info(): ')
    cnt+=1
    print(dframe.info())
  except: pass

  print(f'\n{cnt}: describe(): ')
  cnt+=1
  print(dframe.describe())

  print(f'\n{cnt}: dtypes: ')
  cnt+=1
  print(dframe.dtypes)

  try:
    print(f'\n{cnt}: columns: ')
    cnt+=1
    print(dframe.columns)
  except: pass

  print(f'\n{cnt}: head() -- ')
  cnt+=1
  print(dframe.head())

  print(f'\n{cnt}: shape: ')
  cnt+=1
  print(dframe.shape)

  if (valCnt):
    print('\nValue Counts for each feature -')
    for colname in dframe.columns :
      print(f'\n{cnt}: {colname} value_counts(): ')
      print(dframe[colname].value_counts())
      cnt +=1

# %% 

data = pd.read_csv('okcupidProfiles.csv').dropna()
hobbyData = data.drop(columns=['age','status','sex','orientation','diet','drinks','drugs','education','body_type','ethnicity','height','income','job','last_online','location','offspring','pets','religion','sign','smokes','speaks','essay0','essay1','essay3','essay5','essay6','essay8','essay9'])
dfChk(hobbyData, valCnt=True)

# %%
## pandas method for filtering out words 

dfhobby=data['body_type'].to_frame()
dfhobby['user_ID'] = dfhobby.index
reorder = ['user_ID','body_type']
dfhobby=dfhobby[reorder]

for var in ['essay2','essay4','essay7']:
    dfhobby['Cooking']=hobbyData[var].str.contains('cook', case=False)
    dfhobby['Sports']=hobbyData[var].str.contains('sport', case=False)
    dfhobby['Outdoors']=hobbyData[var].str.contains('outdoors',case=False)
    dfhobby['Yoga']=hobbyData[var].str.contains('yoga',case=False)
    dfhobby['Movies']=hobbyData[var].str.contains('movie',case=False)
    dfhobby['Reading']=hobbyData[var].str.contains('read',case=False)
    dfhobby['Gym']=hobbyData[var].str.contains('gym',case=False)
    dfhobby['Painting']=hobbyData[var].str.contains('paint',case=False)
    dfhobby['VideoGames']=hobbyData[var].str.contains('video game',case=False)
    dfhobby['Bars']=hobbyData[var].str.contains('bar',case=False)
    dfhobby['Dancing']=hobbyData[var].str.contains('danc',case=False)
    dfhobby['Singing']=hobbyData[var].str.contains('sing',case=False)
    dfhobby['Shopping']=hobbyData[var].str.contains('shop',case=False)
    dfhobby['Travel']=hobbyData[var].str.contains('travel',case=False)

# check for and filter out duplicate entries
dfx = dfhobby.copy()
    
def check_dupes(d):
    for x in d:
        if d.duplicated == True:
         print(x)
    else:
        print('No dupes')
            
check_dupes(dfx)

# No duplicates found 

#%% 
## Feature Selection

xfeats = dfx.drop(columns='user_ID')
ytar = dfhobby['body_type'].to_numpy()
ytar=ytar.reshape(-1,1)

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
y = enc.fit_transform(ytar)
X = enc.fit_transform(xfeats)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=1)

from sklearn.feature_selection import SelectKBest, chi2
Xselect = SelectKBest(chi2, k='all')
fs = Xselect.fit(xtrain, ytrain)
Xtrain_fs = Xselect.transform(xtrain)
Xtest_fs = Xselect.transform(xtest)

labels = xfeats.columns

for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores

plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.xlabel('Feature by Column Number')
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], labels = labels,rotation=45)
plt.ylabel('Feature Score')
plt.title('Feature Selection and Importance')
plt.show()

# Features 8, 6, and 12 are the most relevant 
# feature 8 scored 32.618357 = 'VideoGames'
# feature 6 scored 23.971226 = 'Gym'
# feature 12 scored 18.496404 = 'Shopping'


# %%
## KNN Classification

neighbs = 5
y_var = dfhobby['body_type']
x_var = xfeats.drop(columns =['Cooking','Sports','Outdoors','Yoga','Movies','Reading','Painting','Bars','Dancing','Singing','Travel'])

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=neighbs)

# KNN
knn.fit(x_var,y_var)
ypred= knn.predict(x_var)
ypred=knn.predict_proba(x_var)
print(ypred)
print(knn.score(x_var, y))

# KNN-2
knn2 = KNeighborsClassifier(n_neighbors=neighbs)
knn2.fit(xtrain,ytrain)
ytestp = knn2.predict(xtest)
print(knn2.score(xtest,ytest))

# KNN-3
knn3 = KNeighborsClassifier(n_neighbors=neighbs)

from sklearn.model_selection import cross_val_score
crossvalresults = cross_val_score(knn3, x_var, y, cv=10)
print('crossval results:',crossvalresults) 
print(np.mean(crossvalresults)) 

#%%
# Scaled KNN
# 4-KNN algorithm

# Re-do our data with scale on X
from sklearn.preprocessing import scale
xs_var = pd.DataFrame( scale(x_var), columns=x_var.columns ) 
# Note that scale( ) coerce the object from pd.dataframe to np.array  
ys_var = y.copy()  # no need to scale y, but make a true copy / deep copy to be safe
knn_scv = KNeighborsClassifier(n_neighbors=neighbs) # instantiate with n value given
scv_results = cross_val_score(knn_scv, xs_var, ys_var, cv=10)
print(scv_results) 
print(np.mean(scv_results))                

#%%
# Visualiztion

totalvar = xs_var.join(ys_var).dropna()

total = dfx.join(ys_var).dropna()
total = total.drop(columns =['user_ID','Cooking','Sports','Outdoors','Yoga','Movies','Reading','Painting','Bars','Dancing','Singing','Travel'])



#%%
# %%
sns.catplot(x='Gym', y='body_type',data=total)
plt.show()
print(total['Gym'].value_counts(),'\n','proportion of gym-goers = 81/4326 == 1.87%')
sns.catplot(x='VideoGames', y='body_type',data=total)
plt.show()
print(total['VideoGames'].value_counts(),'\n','proportion of video gamers = 70/4337 == 1.61%')
sns.catplot(x='Shopping', y='body_type',data=total)
plt.show()
print(total['Shopping'].value_counts(),'\n','proportion of shoppers = 44/4363 == 1.08%')
# %%
