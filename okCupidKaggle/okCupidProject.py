#%% Trying question 4: 
# Explore a relationship between user defined body type and listed hobbies with k-means clustering 
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

from matplotlib.pyplot import get
import pandas as pd 
import seaborn as sns 
from sklearn.cluster import KMeans

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
hobbyData = data.drop(columns=['age','status','sex','orientation','diet','body_type','drinks','drugs','education','ethnicity','height','income','job','last_online','location','offspring','pets','religion','sign','smokes','speaks','essay0','essay1','essay3','essay5','essay6','essay8','essay9'])
dfChk(hobbyData, valCnt=True)

# %%
## pandas method for filtering out words 

# hobbies = cooking, sports, hiking, yoga, movies, reading, exercise, art, video games, going to a bar, going out with friends

for var in ['essay2','essay4','essay7']:
    cooking = hobbyData.loc[hobbyData[var].str.contains('cook', case=False)]
    sports = hobbyData.loc[hobbyData[var].str.contains('sports', case=False)]
    hiking = hobbyData.loc[hobbyData[var].str.contains('hik', case=False)]
    Yoga = hobbyData.loc[hobbyData[var].str.contains('yoga', case=False)]
    Movies = hobbyData.loc[hobbyData[var].str.contains('Movies', case=False)]
    Reading = hobbyData.loc[hobbyData[var].str.contains('read', case=False)]
    Exercise = hobbyData.loc[hobbyData[var].str.contains('work out', case=False)]
    Art = hobbyData.loc[hobbyData[var].str.contains('art', case=False)]
    VideoGames = hobbyData.loc[hobbyData[var].str.contains('video games', case=False)]
    GoingOut = hobbyData.loc[hobbyData[var].str.contains('going out', case=False)]
    
# pdHobby = pd.concat([cooking, sports, hiking, Yoga,Movies,Reading, Exercise, Art, VideoGames, GoingOut], axis=0,sort=False)
# pdHobby=pdHobby.rename(columns ={'index':'User'}, inplace=True)


# %%
## concatenate dataframes
dfbodyType = data['body_type'].to_frame()
dfbodyType.rename(columns={'index':'User'})

dfhobby = dfbodyType.merge(Art,cooking,Exercise,hiking,)

#%%
## rename index to user 
