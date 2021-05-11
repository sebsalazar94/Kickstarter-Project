"""
REGRESSION : USD_PLEDGED
"""

# (A) Load libraries
import pandas as pd
import numpy as np

# (B) Load data
df=pd.read_excel("Kickstarter.xlsx")

# (C) Drop observations the do not have state ='failed' or 'successful
df=df.loc[(df['state'] == 'failed')|(df['state'] == 'successful')]

# (D) Data cleaning
# Replace nan category with Other
df['category']=df['category'].fillna('Others')

# usd_pledge has extremely high outliers (check boxplot), we will take away the ones higher than the 99th percentile
df=df.loc[df['usd_pledged']<=np.percentile(df['usd_pledged'],99)] 

# (E) Feature engineering
## Turn goal to USD
df['goal_usd']=df['goal']*df['static_usd_rate']

## Determine daily goal during raising campaign
df['goal_usd_perday_launchtodeadline']=df['goal_usd']/df['launch_to_deadline_days']

## Simplify countries by region
df.loc[(df['country'] == 'US')|(df['country'] == 'CA'), 'Region'] = 'North America'
df.loc[(df['country'] == 'CH')|(df['country'] == 'DK')|
       (df['country'] == 'AT')|(df['country'] == 'BE')|
       (df['country'] == 'DE')|(df['country'] == 'ES')|
       (df['country'] == 'FR')|(df['country'] == 'IE')|
       (df['country'] == 'IT')|(df['country'] == 'LU')|
       (df['country'] == 'NL')|(df['country'] == 'GB')|
       (df['country'] == 'NO')|(df['country'] == 'SE'),'Region'] = 'Europe'
df.loc[(df['country'] == 'HK')|(df['country'] == 'SG'), 'Region'] = 'Asia'
df.loc[(df['country'] == 'AU')|(df['country'] == 'NZ'), 'Region'] = 'Oceania'
df.loc[(df['country'] == 'MX'), 'Region'] = 'Latin America'

## Define variables in quarters
df.loc[(df['launched_at_month']>=1)&(df['launched_at_month']<=3),'Launch_Quarter'] = 'Q1'
df.loc[(df['launched_at_month']>=4)&(df['launched_at_month']<=6),'Launch_Quarter'] = 'Q2'
df.loc[(df['launched_at_month']>=7)&(df['launched_at_month']<=9),'Launch_Quarter'] = 'Q3'
df.loc[(df['launched_at_month']>=10)&(df['launched_at_month']<=12),'Launch_Quarter'] = 'Q4'

# (F) Drop variables that are not neccesary because:
### - Do not add value as predictors
df=df.drop(['project_id','name','goal','static_usd_rate','disable_communication'], axis=1)

### - Date are not used as predictors and other variables breakdown their information 
df=df.drop(['deadline','created_at','launched_at',
            'name_len','blurb_len',
            'country'], axis=1)

### - Are invalid predictors
df=df.drop(['state','state_changed_at','state_changed_at_weekday', 'state_changed_at_month',
            'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr', 
            'launch_to_state_change_days',
            'spotlight',## This is done for succesful funded projects
            'staff_pick',## Not present before launch
            'backers_count'], axis=1)

### - Are perfectly correlated with the variable as they are the same
df=df.drop(['pledged'], axis=1)

# (G) Dummify dataframe
df_dummy=pd.get_dummies(df,columns=['Region','currency','category',
                                    'deadline_weekday','created_at_weekday','launched_at_weekday','Launch_Quarter'])

# (H) Outliers removal
from sklearn.ensemble import IsolationForest
iforest=IsolationForest(random_state=0,n_estimators=100, contamination=0.04) ## Contamination goes from 0.0-0.5 it affects what is consider an anomally
pred=iforest.fit_predict(df_dummy)
score=iforest.decision_function(df_dummy)

### Extracting anomalies
from numpy import where
anom_index= where(pred==1)
df_no_outliers= df_dummy.iloc[anom_index]

# (I) Define X and Y
y=df_no_outliers['usd_pledged']
X=df_no_outliers.drop(['usd_pledged'], axis=1)

# (J) Standardize
from sklearn.preprocessing import StandardScaler
standardizer=StandardScaler()
X_std=standardizer.fit_transform(X)

# (K) Feature selection
from sklearn.ensemble import RandomForestRegressor
randomforest=RandomForestRegressor(random_state=0)

#model=randomforest.fit(X,y)
model=randomforest.fit(X_std,y)
model.feature_importances_

feature_rank=pd.DataFrame(list(zip(X.columns,model.feature_importances_*100)), columns = ['predictor','feature importance'])
feature_rank=feature_rank.sort_values(by=['feature importance'],ascending=False)


# (L) Gradient Boosting Modelling

## Determining the optimal number of predictors (from 1 to 85)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
mse_=9999999999999999
optimal_n_features=0

for num_variables in range(1,(len(feature_rank)+1)):
    X_filt_list=[]
    for i in range(num_variables):
        X_filt_list.append(feature_rank.index.tolist()[i])

    X_filt=X_std[:,X_filt_list]
    gbt=GradientBoostingRegressor(random_state=0, n_estimators=100)
    model5=gbt.fit(X_filt, y)
    scores=cross_val_score(estimator=model5,X=X_filt,y=y,cv=5,scoring='neg_mean_squared_error')
    cross_score=np.average(scores)
    if abs(cross_score)<abs(mse_):
        mse_=cross_score
        print('MSE of',num_variables,' features, is lower than',optimal_n_features)
        optimal_n_features=num_variables
        

#####
# Optimal number of predictors is 41
# MSE: 1317825226.38
#####
        
#########################################################################################

"""
CLASSIFICATION : STATE_SUCCESFUL
"""

# (A) Load libraries
import pandas as pd
import numpy as np

# (B) Load data
df=pd.read_excel("Kickstarter.xlsx")

# (C) Drop observations the do not have state ='failed' or 'successful
df=df.loc[(df['state'] == 'failed')|(df['state'] == 'successful')]

# (D) Data cleaning
# Replace nan category with Other
df['category']=df['category'].fillna('Others')

# (E) Feature engineering
## Turn goal to USD
df['goal_usd']=df['goal']*df['static_usd_rate']

## Determine daily goal during raising campaign
df['goal_usd_perday_launchtodeadline']=df['goal_usd']/df['launch_to_deadline_days']

## Simplify countries by region
df.loc[(df['country'] == 'US')|(df['country'] == 'CA'), 'Region'] = 'North America'
df.loc[(df['country'] == 'CH')|(df['country'] == 'DK')|
       (df['country'] == 'AT')|(df['country'] == 'BE')|
       (df['country'] == 'DE')|(df['country'] == 'ES')|
       (df['country'] == 'FR')|(df['country'] == 'IE')|
       (df['country'] == 'IT')|(df['country'] == 'LU')|
       (df['country'] == 'NL')|(df['country'] == 'GB')|
       (df['country'] == 'NO')|(df['country'] == 'SE'),'Region'] = 'Europe'
df.loc[(df['country'] == 'HK')|(df['country'] == 'SG'), 'Region'] = 'Asia'
df.loc[(df['country'] == 'AU')|(df['country'] == 'NZ'), 'Region'] = 'Oceania'
df.loc[(df['country'] == 'MX'), 'Region'] = 'Latin America'

## Define variables in quarters
df.loc[(df['launched_at_month']>=1)&(df['launched_at_month']<=3),'Launch_Quarter'] = 'Q1'
df.loc[(df['launched_at_month']>=4)&(df['launched_at_month']<=6),'Launch_Quarter'] = 'Q2'
df.loc[(df['launched_at_month']>=7)&(df['launched_at_month']<=9),'Launch_Quarter'] = 'Q3'
df.loc[(df['launched_at_month']>=10)&(df['launched_at_month']<=12),'Launch_Quarter'] = 'Q4'

# (F) Drop variables that are not neccesary because:
### - Do not add value as predictors
df=df.drop(['project_id','name','goal','static_usd_rate','disable_communication'], axis=1)

### - Date are not used as predictors and other variables breakdown their information 
df=df.drop(['deadline','created_at','launched_at',
            'name_len','blurb_len',
            'country'], axis=1)

### - Are invalid predictors
df=df.drop(['pledged','usd_pledged','state_changed_at','state_changed_at_weekday', 'state_changed_at_month',
            'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr', 
            'launch_to_state_change_days',
            'spotlight',## This is done for succesful funded projects
            'staff_pick',## Not present before launch
            'backers_count'], axis=1)


# (G) Dummify dataframe
df_dummy=pd.get_dummies(df,columns=['Region','currency','category',
                                    'deadline_weekday','created_at_weekday','launched_at_weekday','Launch_Quarter',
                                    'state'])


# (H) Define X and Y
y=df_dummy['state_successful']
X=df_dummy.drop(['state_successful','state_failed'], axis=1)

# (I) Standardize
from sklearn.preprocessing import StandardScaler
standardizer=StandardScaler()
X_std=standardizer.fit_transform(X)

# (J) Feature selection
from sklearn.ensemble import RandomForestClassifier
randomforest=RandomForestClassifier(random_state=0)

model=randomforest.fit(X_std,y)
model.feature_importances_

feature_rank2=pd.DataFrame(list(zip(X.columns,model.feature_importances_*100)), columns = ['predictor','feature importance'])
feature_rank2=feature_rank2.sort_values(by=['feature importance'],ascending=False)

# (K) Gradient Boosting Modelling

## Determining the optimal number of predictors (from 1 to 85)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
accuracy=0
optimal_n_features=0

for num_variables in range(1,(len(feature_rank)+1)):
    X_filt_list=[]
    for i in range(num_variables):
        X_filt_list.append(feature_rank2.index.tolist()[i])

    X_filt=X_std[:,X_filt_list]
    gbt=GradientBoostingClassifier(random_state=0)
    model=gbt.fit(X_filt, y)
    scores=cross_val_score(estimator=model,X=X_filt,y=y,cv=5)
    cross_score=np.average(scores)
    if cross_score>accuracy:
        accuracy=cross_score
        print('Accuracy of',num_variables,' features, is higher than',optimal_n_features,'- ',accuracy)
        optimal_n_features=num_variables

#####
# Optimal number of predictors is 65
# Accuracy: 0.752183
#####
        
#########################################################################################

"""
CLUSTERING MODEL
"""

# (A) Load libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# (B) Load data
df=pd.read_excel("Kickstarter.xlsx")

# (C) Drop observations the do not have state ='failed' or 'successful
df=df.loc[(df['state'] == 'failed')|(df['state'] == 'successful')]

# (D) Feature engineering
## Turn goal to USD
df['goal_usd']=df['goal']*df['static_usd_rate']

# (E) Keep only three numerical variables
#X=df[['usd_pledged','goal_usd','create_to_launch_days']] # Numericals from regression
X=df[['usd_pledged','goal_usd','backers_count']]

# (F) Remove outliers with isolation forests
from sklearn.ensemble import IsolationForest
iforest=IsolationForest(random_state=0,n_estimators=100, contamination=0.04) ## Contamination goes from 0.0-0.5 it affects what is consider an anomally
pred=iforest.fit_predict(X)
score=iforest.decision_function(X)

### Extracting anomalies
from numpy import where
anom_index= where(pred==1)
X_no_outliers= X.iloc[anom_index]

# (G) Standardize
from sklearn.preprocessing import StandardScaler
standardizer=StandardScaler()
X_std=standardizer.fit_transform(X_no_outliers)

# (H) Clustering with K Means
# Finding the optimal number of clusters (Elbow method)
from sklearn.cluster import KMeans
withinss=[]
for i in range(2,8):
    kmeans=KMeans(n_clusters=i)
    model=kmeans.fit(X_std)
    withinss.append(model.inertia_)

from matplotlib import pyplot
pyplot.plot([2,3,4,5,6,7],withinss)
 
# Optimal n_clusters=4   

# Running KMeans with optimal K
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4)
model=kmeans.fit(X_std)
labels=model.predict(X_std)
    
# Plot the clustering in 3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import pyplot

xyz_order=[1,2,0]

fig = plt.figure(figsize=(10,6))
cluster_plot= fig.gca(projection='3d')
cluster_plot.scatter(X_no_outliers.iloc[:,xyz_order[0]]/100000, X_no_outliers.iloc[:,xyz_order[1]],X_no_outliers.iloc[:,xyz_order[2]]/100000 ,
                     c=labels, cmap='rainbow')
cluster_plot.set_xlabel('Goal (US Millions)')
cluster_plot.set_ylabel('Number of backers')
cluster_plot.set_zlabel('Pledged (US Millions)')