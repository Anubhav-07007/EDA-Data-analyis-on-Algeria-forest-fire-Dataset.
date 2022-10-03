#!/usr/bin/env python
# coding: utf-8

# # EDA & Data analyis on Algeria forest fire Dataset.
# # Data Set Information:
# 
# The dataset includes 244 instances that regroup a data of two regions of Algeria,namely the Bejaia region located in the northeast of Algeria and the Sidi Bel-abbes region located in the northwest of Algeria.
# 
# 122 instances for each region.
# 
# The period from June 2012 to September 2012.
# The dataset includes 11 attribues and 1 output attribue (class)
# 

# # Attribute Information:
# 
# 1. Date : (DD/MM/YYYY) Day, month ('june' to 'september'), year (2012)
# 
# # Weather data observations
# 2. Temp : temperature noon (temperature max) in Celsius degrees: 22 to 42
# 3. RH : Relative Humidity in %: 21 to 90
# 4. Ws :Wind speed in km/h: 6 to 29
# 5. Rain: total day in mm: 0 to 16.8
# 
# # FWI Components
# 6. Fine Fuel Moisture Code (FFMC) index from the FWI system: 28.6 to 92.5
# 7. Duff Moisture Code (DMC) index from the FWI system: 1.1 to 65.9
# 8. Drought Code (DC) index from the FWI system: 7 to 220.4
# 9. Initial Spread Index (ISI) index from the FWI system: 0 to 18.5
# 10. Buildup Index (BUI) index from the FWI system: 1.1 to 68
# 11. Fire Weather Index (FWI) Index: 0 to 31.1
# 12. Classes: two classes, namely Fire and not Fire

# # Importing All the required Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
from six.moves import urllib

warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Data reading | head and tail of data. 

df=pd.read_csv(r'C:\Users\anubh\Downloads\Algerian_forest_fires_dataset_UPDATE.csv',header=1)

df


# # Basic Information about Data | Set
# 
# 1.  Column & Rows 
# 2.  Missing value
# 3.  Datatype
# 

# In[3]:


df.info()


# # Find out rows where no information.

# In[4]:


df.loc[122:].head()     # find the row where we don't have data. 

df.drop(index=[122,123],inplace=True)  # Dropping the row 
df.reset_index(inplace=True)            # Reset the index value


# # Data set divided into 2 group Bejaia Region &  Sidi Bel-abbes Region 
# # We have 122 records for each group. 
# # Let's create region column.

# In[5]:


df['Region']=0

for i in range(len(df)):
    if i <122:
        df['Region'][i]=0
    else:
        df['Region'][i]=1
        
df


# In[6]:


# Lets cross check the output...

l1=[]
l2=[]

for i in df['Region']:
    if i==0:
        l1.append(i)
    else:
        l2.append(i)
print(len(l1))
print(len(l2))        # Looks like the record assinged equally & Correctly. 


# In[7]:


df.columns


# # Dataset columns have some extra space in column. 
# # Let's remove it. 

# In[8]:


for col in df.columns:
    df.rename(columns={col:col.strip()},inplace=True)
    
print(df.columns)        # Now its looks fine. 


# # Record at mis-match postion. Let's correct it 

# In[9]:


df.iloc[165]


# In[10]:


df.at[165,'DC']=14.6
df.at[165,'ISI']=9
df.at[165,'BUI']=12.5
df.at[165,'FWI']=10.4
df.at[165,'Classes']='fire'


# In[11]:


df.loc[165]  # its looks fine now


# # checking the data type of each column and change it correctly. 

# In[12]:


df.info()


# In[13]:


df[['day','month','year','Temperature','RH','Ws']]=df[['day','month','year','Temperature','RH','Ws']].astype('int')
df[['Rain','FFMC','DMC','DC','ISI','BUI','FWI']]=df[['Rain','FFMC','DMC','DC','ISI','BUI','FWI']].astype('float')


# In[14]:


# 'day','month','year','Temperature','RH','Ws' We have continous variable without decimal. converted into int64
# 'Rain','FFMC','DMC','DC','ISI','BUI','FWI'. We have continous variable with decimal. converted into float

df.info()


# # Check how many unique categories we in Classes column. 

# In[15]:


df['Classes'].unique()


# In[16]:


# We have extra spaces in categories. let's fix this out. 

df['Classes']=[i.strip()  for i in df['Classes']]

# let's check now. 

df['Classes'].unique()     # its fine now. 


# # Shape of data

# In[17]:


df.drop(['index'],axis=1,inplace=True)
df.shape


# # Summary of data

# In[18]:


df.describe().T


# # Missing vlaue check

# In[19]:


df.isnull().sum()


# # Find : How many numeric & categorical feature in dataset. 

# # Numeric Feature

# In[20]:


numeric_features=[feature for feature in df.columns if df[feature].dtypes!='object']

print('We have total {} numeric feature and the feature is : {}'.format(len(numeric_features),numeric_features))


# # Categorical Feature

# In[21]:


categorical_feature=[feature for feature in df.columns if df[feature].dtype=='object']
print('We have total {} categorical feature and the feature is : {}'.format(len(categorical_feature),categorical_feature))


# # Discreate Feature

# In[22]:


discreate_feature=[feature for feature in numeric_features if len(df[feature].unique())<35]
print('We have total {} discreate_feature and the feature is : {}'.format(len(discreate_feature),discreate_feature))


# # Continous numeric Features

# In[23]:


continous_numeric_features=[feature for feature in numeric_features if feature not in discreate_feature]
print('We have total {} continous numeric feature and feature is {}'.format(len(continous_numeric_features),continous_numeric_features))


# # Univariate analysis 

# In[24]:


for feature in continous_numeric_features:
    sns.histplot(data=df,x=feature,kde=True,bins=15,color='green')
    plt.show()


# # Observation on univariate 
# 
# 1. Relative Humidity is following gaussian distribution. 
# 2. Rain , DMC , DC, ISI , BUI , FWI are following right skewed distribution(log-Normal-distribution)
# 3. FFMC feature following left skewed distribution. 

# In[25]:


sns.countplot(df['Classes'])


# # Bivariate analyis between discreate numerical feature and target feature

# In[26]:


for i in categorical_feature:
    print(df.groupby(i)['Region'].value_counts())


# In[27]:


sns.countplot(data=df,x='Classes',hue='Region')


# In[28]:


for feature in discreate_feature:
    sns.countplot(data=df,x=feature,hue='Classes')
    plt.show()


# #  Obersvation 
# 
# 1. Day vs Classes : almost everday the occurance of fire is visible, and the count of fire is greater or equal to not fire count.
# 2. Month vs Classes : Occurance of fire is high in July and August, as compare to June and september. 
# 3. Year vs Classes : Occurance of fire is high in 2012.
# 4. Temperature vs Classes : If the Temperature in between in 36 to 37. then there is high chances for fire occured. 
# 5. Ws(Wind speed) vs Classes : if the Wind speed in between 13 to 19. then there is high chances for fire occured.
# 6. Region vs Classes : Sidi Bel abbes region has more fire casses.

# In[29]:


for feature in continous_numeric_features:
    sns.boxplot(data=df,x=feature,color='g')
    plt.title(feature)
    plt.show()


# # Obesrvation 
# 
#  Feature having outlier : Rain , FFMC ,DMC , DC, ISI, BUI, FWI

# In[30]:


data=df[[feature for feature in numeric_features if feature not in ['day', 'month', 'year', 'Temperature', 'Ws', 'Region']]].corr()


# In[31]:


plt.figure(figsize=(15, 15))
sns.heatmap(data)


# # Heat map obesrvation
# 
# 1. correlation coefficients between 0.8 to to 1 | very high correlated.
# 2. correlation coefficients between 0.6 to 0.8  | high correlated.
# 3. correlation coefficients between 0.4 to 0.6  | modreate correlated. 
# 4. correlation coefficients between 0.2 to 0.4  | less correalted. 
# 
# Very High Correlated :  DMC-BUI , DC-BUI , FWI- BUI
#     High correlated  :  FFMC-BUI , ISI-BUI, DC-ISI
#    

# # Final Report 
# 
# Observation on univariate analysis
# 
# 1.Relative Humidity is following gaussian distribution.
# 2.Rain , DMC , DC, ISI , BUI , FWI are following right skewed distribution(log-Normal-distribution)
# 3.FFMC feature following left skewed distribution.
# 
# Obersvation on bivariate analysis 
# 
# 1. Day vs Classes : almost everday the occurance of fire is visible, and the count of fire is greater or equal to not fire count.
# 2. Month vs Classes : Occurance of fire is high in July and August, as compare to June and september. 
# 3. Year vs Classes : Occurance of fire is high in 2012.
# 4. Temperature vs Classes : If the Temperature in between in 36 to 37. then there is high chances for fire occured. 
# 5. Ws(Wind speed) vs Classes : if the Wind speed in between 13 to 19. then there is high chances for fire occured.
# 6. Region vs Classes : Sidi Bel abbes region has more fire casses.
# 
# Obesrvation on Outlier 
# 
# Feature having outlier : Rain , FFMC ,DMC , DC, ISI, BUI, FWI
# 
# Obesrvation Heat map
# 
# 1. correlation coefficients between 0.8 to to 1 | very high correlated.
# 2. correlation coefficients between 0.6 to 0.8  | high correlated.
# 3. correlation coefficients between 0.4 to 0.6  | modreate correlated. 
# 4. correlation coefficients between 0.2 to 0.4  | less correalted. 
# 
# Very High Correlated :  DMC-BUI , DC-BUI , FWI- BUI
#     High correlated  :  FFMC-BUI , ISI-BUI, DC-ISI

# In[ ]:




