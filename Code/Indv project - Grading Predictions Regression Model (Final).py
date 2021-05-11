"""
KICKSTARTER GRADING - PREDICT CODE
 
"""

"""
(1) - REGRESSION PREDICTIONS
 
"""

import pandas as pd
import numpy as np

## (A) Train the model  #################

# Load data
df=pd.read_excel("Kickstarter.xlsx")

# Drop observations the do not have state ='failed' or 'successful
df=df.loc[(df['state'] == 'failed')|(df['state'] == 'successful')]

# Data cleaning
# Replace nan category with Other
df['category']=df['category'].fillna('Others')

# usd_pledge has extremely high outliers (check boxplot), we will take away the ones higher than the 99th percentile
df=df.loc[df['usd_pledged']<=np.percentile(df['usd_pledged'],99)] 

# Feature engineering
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

# Drop variables that are not neccesary because:
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

# Dummify dataframe
df_dummy=pd.get_dummies(df,columns=['Region','currency','category',
                                    'deadline_weekday','created_at_weekday','launched_at_weekday','Launch_Quarter'])

# Outliers removal
from sklearn.ensemble import IsolationForest
iforest=IsolationForest(random_state=0,n_estimators=100, contamination=0.04) ## Contamination goes from 0.0-0.5 it affects what is consider an anomally
pred=iforest.fit_predict(df_dummy)
score=iforest.decision_function(df_dummy)

### Extracting anomalies
from numpy import where
anom_index= where(pred==1)
df_no_outliers= df_dummy.iloc[anom_index]

# Define X and Y
y=df_no_outliers['usd_pledged']
X=df_no_outliers.drop(['usd_pledged'], axis=1)

# Standardize
from sklearn.preprocessing import StandardScaler
standardizer=StandardScaler()
X_std=standardizer.fit_transform(X)

# Feature selection
from sklearn.ensemble import RandomForestRegressor
randomforest=RandomForestRegressor(random_state=0)


model=randomforest.fit(X_std,y)
model.feature_importances_

feature_rank=pd.DataFrame(list(zip(X.columns,model.feature_importances_*100)), columns = ['predictor','feature importance'])
feature_rank=feature_rank.sort_values(by=['feature importance'],ascending=False)

num_variables=41
X_filt_list=[]
for i in range(num_variables):
    X_filt_list.append(feature_rank.index.tolist()[i])

# Final model train
from sklearn.ensemble import GradientBoostingRegressor
X_filt=X_std[:,X_filt_list]
gbt=GradientBoostingRegressor(random_state=0, n_estimators=100)
model5=gbt.fit(X_filt, y)


## (B) Predicting  #################

df_g=pd.read_excel("Kickstarter-Grading-Sample.xlsx")

# Drop observations the do not have state ='failed' or 'successful
df_g=df_g.loc[(df_g['state'] == 'failed')|(df_g['state'] == 'successful')]

# Data cleaning
# Replace nan category with Other
df_g['category']=df_g['category'].fillna('Others')


# Feature engineering
## Turn goal to USD
df_g['goal_usd']=df_g['goal']*df_g['static_usd_rate']

## Determine daily goal during raising campaign
df_g['goal_usd_perday_launchtodeadline']=df_g['goal_usd']/df_g['launch_to_deadline_days']

## Simplify countries by region
df_g.loc[(df_g['country'] == 'US')|(df_g['country'] == 'CA'), 'Region'] = 'North America'
df_g.loc[(df_g['country'] == 'CH')|(df_g['country'] == 'DK')|
       (df_g['country'] == 'AT')|(df_g['country'] == 'BE')|
       (df_g['country'] == 'DE')|(df_g['country'] == 'ES')|
       (df_g['country'] == 'FR')|(df_g['country'] == 'IE')|
       (df_g['country'] == 'IT')|(df_g['country'] == 'LU')|
       (df_g['country'] == 'NL')|(df_g['country'] == 'GB')|
       (df_g['country'] == 'NO')|(df_g['country'] == 'SE'),'Region'] = 'Europe'
df_g.loc[(df_g['country'] == 'HK')|(df_g['country'] == 'SG'), 'Region'] = 'Asia'
df_g.loc[(df_g['country'] == 'AU')|(df_g['country'] == 'NZ'), 'Region'] = 'Oceania'
df_g.loc[(df_g['country'] == 'MX'), 'Region'] = 'Latin America'

## Define variables in quarters
df_g.loc[(df_g['launched_at_month']>=1)&(df_g['launched_at_month']<=3),'Launch_Quarter'] = 'Q1'
df_g.loc[(df_g['launched_at_month']>=4)&(df_g['launched_at_month']<=6),'Launch_Quarter'] = 'Q2'
df_g.loc[(df_g['launched_at_month']>=7)&(df_g['launched_at_month']<=9),'Launch_Quarter'] = 'Q3'
df_g.loc[(df_g['launched_at_month']>=10)&(df_g['launched_at_month']<=12),'Launch_Quarter'] = 'Q4'

# Drop variables that are not neccesary because:
### - Do not add value as predictors
df_g=df_g.drop(['project_id','name','goal','static_usd_rate','disable_communication'], axis=1)

### - Date are not used as predictors and other variables breakdown their information 
df_g=df_g.drop(['deadline','created_at','launched_at',
            'name_len','blurb_len',
            'country'], axis=1)

### - Are invalid predictors
df_g=df_g.drop(['state','state_changed_at','state_changed_at_weekday', 'state_changed_at_month',
            'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr', 
            'launch_to_state_change_days',
            'spotlight',## This is done for succesful funded projects
            'staff_pick',## Not present before launch
            'backers_count'], axis=1)

### - Are perfectly correlated with the variable as they are the same
df_g=df_g.drop(['pledged'], axis=1)

df_dummy_g=pd.get_dummies(df_g,columns=['Region','currency','category',
                                    'deadline_weekday','created_at_weekday','launched_at_weekday','Launch_Quarter'])

# Define X and Y
y_g=df_dummy_g['usd_pledged']
X_g=df_dummy_g.drop(['usd_pledged'], axis=1)

# Keep variables from the final model
num_variables=41
X_filt_list_g=[]
for i in range(num_variables):
    X_filt_list_g.append(feature_rank['predictor'].tolist()[i])

X_g=X_g[X_filt_list_g]

# Standardize
from sklearn.preprocessing import StandardScaler
standardizer=StandardScaler()
X_std_g=standardizer.fit_transform(X_g)


## Predict
y_g_pred = model5.predict(X_std_g)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_g_pred, y_g)
print('MSE for Grading database is : '+ str(mse))
print('Warning, this model is not taking outliers away, for prediction consistency, MSE can be greatly improved without this outliers')