#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[82]:


df_21 = pd.read_csv('male_players (legacy).csv')


# In[83]:


df_22 = pd.read_csv('players_22.csv')


# In[91]:


df_21.info()


# In[ ]:


# percentage of missing data in each column
missing_percentage21 = df_21.isnull().mean()*100
missing_percentage22 = df_22.isnull().mean()*100


# In[89]:


for col in df_21.columns:
    print(f'{col}:{df_21[col].isnull().sum()}')


# In[92]:


#Index of columns whose missing data threshold is greater than 50%
columns_to_drop21 = missing_percentage21[missing_percentage21 > 30].index
columns_to_drop22 = missing_percentage22[missing_percentage22 > 30].index


# In[98]:


columns_to_drop21


# In[ ]:


df_21 = df_21.drop(columns=columns_to_drop21)
df_22 = df_22.drop(columns=columns_to_drop22)


# In[102]:


for col in df_21.columns:
    print(col)


# In[139]:


#removing non essential columns
columns_to_drop21 = [ 'player_url', 'short_name', 'long_name', 'dob',
                   'club_team_id',  'nationality_name','real_face',
                   'player_face_url']

columns_to_drop22 = ['sofifa_id', 'player_url', 'short_name', 'long_name', 'dob',
                   'club_team_id', 'nationality_name','real_face',
                   'player_face_url', 'club_logo_url', 'club_flag_url', 'nation_flag_url']

df_21_dropped = df_21.drop(columns=columns_to_drop21, axis=1)
df_22_dropped = df_22.drop(columns=columns_to_drop22, axis=1)


# In[ ]:





# In[140]:


#checking if missing values actually do not exceed threshold
with open('missing_values.txt','w') as f:
    for col in df_21_dropped.columns:
        missing = df_21_dropped[col].isnull().sum()
        if missing > 0:
            f.write(f"{col}:{missing}\n")
    f.write("\n\n\n")
    for col in df_22_dropped.columns:
        missing = df_22_dropped[col].isnull().sum()
        if missing > 0:
            f.write(f"{col}:{missing}\n")
    


# In[141]:


#seperating numerical data from non nunmeric data
numerical_data21 = df_21_dropped.select_dtypes(np.number)
numerical_data22 = df_22_dropped.select_dtypes(np.number)
categorical_data21 = df_21_dropped.select_dtypes(['object'])
categorical_data22 = df_22_dropped.select_dtypes(['object'])


# In[142]:


numerical_data21


# In[143]:


numerical_data22


# In[144]:


#Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=1, random_state=0)


# In[145]:


imp.fit(numerical_data21)
imp.fit(numerical_data22)


# In[146]:


imputed_numerical_data21 = np.round(imp.fit_transform(numerical_data21))
imputed_numerical_data22 = np.round(imp.fit_transform(numerical_data22))


# In[147]:


imputed_numerical_data21


# In[148]:


#imputed data
numerical_21 = pd.DataFrame(imputed_numerical_data21, columns=numerical_data21.columns)
numerical_22 = pd.DataFrame(imputed_numerical_data22, columns=numerical_data22.columns)


# In[149]:


numerical_21


# In[150]:


#confirming success of imputation
with open('missing_values.txt','w') as f:
    for col in numerical_21.columns:
        missing = numerical_21[col].isnull().sum()
        f.write(f"{col}:{missing}\n")
    f.write("\n\n\n")
    for col in numerical_22.columns:
        missing = numerical_22[col].isnull().sum()
        f.write(f"{col}:{missing}\n")


# In[152]:


#Checking for missing categorical values
with open('categorical.txt', 'w') as f:
    for col in categorical_data21.columns:
        missing = categorical_data21[col].isnull().sum()
        f.write(f"{col}: {missing} \n")
    f.write('\n\n\n')
    for col in categorical_data22.columns:
        missing = categorical_data22[col].isnull().sum()
        f.write(f"{col}: {missing}\n")


# In[153]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')


# In[154]:


imputed_categorical21 = imputer.fit_transform(categorical_data21)
imputed_categorical22 = imputer.fit_transform(categorical_data22)


# In[155]:


categorical_21 = pd.DataFrame(imputed_categorical21, columns=categorical_data21.columns)
categorical_22 = pd.DataFrame(imputed_categorical22, columns=categorical_data22.columns)


# In[156]:


#confirming success of categorical data imputation
with open('categorical.txt', 'w') as f:
    for col in categorical_21.columns:
        missing = categorical_21[col].isnull().sum()
        f.write(f"{col}: {missing} \n")
    f.write('\n\n\n')
    for col in categorical_22.columns:
        missing = categorical_22[col].isnull().sum()
        f.write(f"{col}: {missing}\n")


# In[157]:


total_instances21 = categorical_21.shape[0]
total_instances22 = categorical_22.shape[0]


# In[164]:


#setting threshold of occurances of data in column to 1% of dataset
threshold_21 = total_instances21 *  0.01
threshold_22 = total_instances22 * 0.01
print(f"minimum threshold is {threshold_21}.")
print(f"minimum threshold is {threshold_22}.")


# In[165]:


cat_21 = categorical_21.apply(lambda x: x.mask(x.map(x.value_counts()) < threshold_21, 'rare') if x.name in categorical_21.columns else x)
cat_22 = categorical_22.apply(lambda x: x.mask(x.map(x.value_counts()) < threshold_22, 'rare') if x.name in categorical_22.columns else x)


# In[166]:


#Encoding the categorical data
categorical_data21 = pd.get_dummies(cat_21)
categorical_data22 = pd.get_dummies(cat_22)
categorical_data21 = categorical_data21.astype(int)
categorical_data22 = categorical_data22.astype(int)


# In[167]:


categorical_data22


# In[168]:


#joinig the numerical and categorical dataframes
dframe_21 = pd.concat([numerical_21, categorical_data21], axis=1)
dframe_22 = pd.concat([numerical_22, categorical_data22], axis=1)


# In[171]:


dframe_22.info()


# In[172]:


corr_matrix21 = dframe_21.corr()


# In[173]:


corr_matrix22 = dframe_22.corr()


# In[174]:


#Dependent variable
y_21 = dframe_21['overall']
y_22= dframe_22['overall']


# In[27]:


#Adding the dependent variable to the categorical dataframe to determine the strength of the correlation
temp_21 = pd.concat([y_21,categorical_data21], axis=1)
temp_22 = pd.concat([y_22,categorical_data22], axis=1)


# In[175]:


dframe_21.info()


# In[28]:


temp_21_corr = temp_21.corr()


# In[29]:


temp_22_corr = temp_22.corr()


# In[176]:


#checking the correlation strength of the dependent variable with the data
corr_21 = corr_matrix21['overall'].abs().sort_values(ascending=False)


# In[177]:


corr_22 = corr_matrix22['overall'].abs().sort_values(ascending=False)


# In[224]:


corr_21.head(30)


# In[179]:


corr_22.head(30)


# In[188]:


#index of numerical with maximum correlation with dependent variable
selected_features21 = corr_21[corr_21 > 0.4].index
selected_features22 = corr_22[corr_22 > 0.4].index


# In[226]:


selected_features21


# In[191]:


#dataframe of feature variables with maximum correlation
new_df21 = dframe_21[selected_features21]
new_df22 = dframe_22[selected_features22]


# In[192]:


#target variable
y_21 = new_df21['overall']
y_22 = new_df22['overall']


# In[193]:


X_21 = new_df21.drop(columns=['overall'])
X_22 = new_df22.drop(columns=['overall'])


# In[194]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[195]:


#Scaling feature variables
X_21_scaled = sc.fit_transform(X_21)
X_22_scaled = sc.fit_transform(X_22)


# In[196]:


X_21= pd.DataFrame(X_21_scaled, columns=X_21.columns)


# In[197]:


X_22 = pd.DataFrame(X_22_scaled, columns=X_22.columns)


# In[198]:


from sklearn.model_selection import train_test_split, cross_val_score


# In[199]:


X_21train, X_21test, y_21train, y_21test = train_test_split(X_21, y_21, test_size=0.2, random_state=42)


# In[200]:


X_21train.shape


# In[201]:


X_21test.shape


# In[ ]:



# In[202]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


# In[203]:


#Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
cross_vals = cross_val_score(rf_model, X_21train, y_21train,  cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cross_vals.mean())


# In[204]:


print(cv_rmse)


# In[205]:


rf_model.fit(X_21train, y_21train)


# In[256]:


y_21pred = rf_model.predict(X_21test)


# In[257]:


from sklearn.metrics import mean_squared_error,r2_score


# In[258]:


print(f"""Random Forest Regressor Mean Squared error ={mean_squared_error(y_21pred,y_21test)}
R2 Score: {r2_score(y_21pred, y_21test)}""") 


# In[209]:


#Gradient boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
cross_vals = cross_val_score(gb_model, X_21train, y_21train,  cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cross_vals.mean())


# In[210]:


print(cv_rmse)


# In[211]:


gb_model.fit(X_21train, y_21train)


# In[259]:


y_21pred = gb_model.predict(X_21test)


# In[260]:


print(f""" Gradient Boosting Regressor Mean Squared error={mean_squared_error(y_21pred, y_21test)}
R2 Score: {r2_score(y_21pred, y_21test)}""") 


# In[ ]:





# In[214]:


#xgb boost regressor model
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
cross_vals = cross_val_score(xgb_model, X_21train, y_21train,  cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cross_vals.mean())


# In[215]:


print(cv_rmse)


# In[216]:


xgb_model.fit(X_21train, y_21train)


# In[261]:


y_21pred = xgb_model.predict(X_21test)


# In[262]:


print(f"""XGB Regressor Mean Squared error={mean_squared_error(y_21pred, y_21test)},
R2 Score: {r2_score(y_21pred, y_21test)}""") 


# In[219]:


#Picking the best model
best_model = rf_model


# In[220]:


best_model


# In[241]:


#Ensuring that both datsets have the same features

for col in X_22.columns:
    if col not in X_21.columns:
        X_22.drop(col,axis=1,inplace=True)


# In[243]:


select_features=[]
for col in X_21.columns:
    select_features.append(col)


# In[245]:


X_22 = X_22[select_features]


# In[246]:


X_22.columns


# In[247]:


X_21.columns


# In[263]:


y_22pred = best_model.predict(X_22)


# In[264]:


# Evaluate the model's performance on the 2022 data
mse_22 = mean_squared_error(y_22pred, y_22)


# In[265]:


print(f"""Mean square error with fifa 22 dataset = {mse_22}, 
R2 Score: {r2_score(y_22pred, y_22)}""")


# In[266]:


import pickle as pkl


# In[267]:


#writing model to file
with open('best_model.pkl', 'wb') as f:
    pkl.dump(best_model,f)


# In[268]:


#loading model from file
with open('best_model.pkl','rb') as file:
    loaded_model = pkl.load(file)


# In[269]:


#Testing loaded model
y_pred = loaded_model.predict(X_22)


# In[270]:


print(f"""Mean square error with fifa 22 dataset = {mean_squared_error(y_pred,y_22)}, 
R2 Score: {r2_score(y_pred, y_22)}""")





