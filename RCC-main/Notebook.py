#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import logging

df = pd.read_csv('speed-dating_csv.csv')
df=df[['age','gender','race','importance_same_religion','field','reading','gaming', 'cleanliness','music','movies','tv','clubbing','hiking','exercise','sports','tvsports','d_importance_same_religion','d_reading','d_gaming', 'd_cleanliness','d_music','d_movies','d_tv','d_clubbing','d_hiking','d_exercise','d_sports','d_tvsports']]
df


# In[2]:


def normalize_values(df,current_feature_name ,current_scaled_feature_name , new_feature_name):
    # current_feature_name = 'yoga'
    # current_scaled_feature_name = 'd_yoga'
    # new_feature_name = 'yoga_normalized'
    temp = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    temp[new_feature_name] = 0  # Add a new column to store normalized values
    
    for index, row in temp.iterrows():
        param = row[current_feature_name]
        scale_param = int(row[current_scaled_feature_name].split(']')[0].split('-')[-1])
        param_normalized = param / scale_param
        temp.at[index, new_feature_name] = param_normalized
    
    return temp

df = normalize_values(df,'reading','d_reading','reading_normalized')
df = normalize_values(df,'gaming','d_gaming','gaming_normalized')
df = normalize_values(df,'cleanliness','d_cleanliness','cleanliness_normalized')
df = normalize_values(df,'music','d_music','music_normalized')
df = normalize_values(df,'movies','d_movies','movies_normalized')
df = normalize_values(df,'tv','d_tv','tv_normalized')
df = normalize_values(df,'clubbing','d_clubbing','clubbing_normalized')
df = normalize_values(df,'hiking','d_hiking','hiking_normalized')
df = normalize_values(df,'exercise','d_exercise','exercise_normalized')
df = normalize_values(df,'sports','d_sports','sports_normalized')
df = normalize_values(df,'importance_same_religion','d_importance_same_religion','importance_same_religion_normalized')

interestedColumns = ['age','gender','race','importance_same_religion','field','reading','gaming',
                 'cleanliness','music','movies','tv','clubbing','hiking','exercise',
                 'sports','tvsports','reading_normalized',
                 'gaming_normalized','cleanliness_normalized','music_normalized',
                 'movies_normalized','tv_normalized','clubbing_normalized',
                 'hiking_normalized','exercise_normalized','sports_normalized',
                 'importance_same_religion_normalized']
df = df[interestedColumns]
df


# In[3]:


def measure_age_range(value):
    age_bins = [(15,20), (21,25),(26,30),(31,35),(36,40),(41,60)]
    for range in age_bins:
        min_age = range[0]
        max_age = range[1]
        if (value >= min_age) and (value <= max_age):
            return range

df['age-range'] = df['age'].apply(measure_age_range)
# Adding a random 'FoodChoice' column with binary values
df['FoodChoice'] = np.random.choice([0, 1], size=len(df))
df['Smoking'] = np.random.choice([0, 1,2], size=len(df))
df['Drinking'] = np.random.choice([0, 1,2], size=len(df))



df['Smoking'] = df['Smoking'].apply(lambda food: 'No' if food == 0 else ('Yes' if food == 1  else 'Sometimes'))
df['Drinking'] = df['Drinking'].apply(lambda food: 'No' if food == 0 else ('Yes' if food == 1  else 'Occasionally'))

# Converting binary values to labels
df['FoodChoice'] = df['FoodChoice'].apply(lambda food: 'Vegetarian' if food == 0 else 'Non-Vegetarian')

listSubdivOfRegina = ['Albert Park',
'Arcola Sub',
'Argyle Park',
'Arnhem Place',
'Assiniboia East',
'Assiniboia Place',
'Balbrigan Place',
'Belvedere',
'Broders Annex',
'Churchill Place',
'City View',
'Connaught Park',
'Coronation Park',
'Coventry Place',
'CPR Annex',
'Creekside',
'Crescents',
'Dewdney Place',
'Dieppe Place',
'Dominion Heights',
'Douglas Park',
'Douglas Place',
'Eastern Annex',
'Eastgate',
'Eastpoint Estates',
'Fairways West',
'Garden Ridge',
'Gardiner Heights',
'Glencairn',
'Glencairn Village',
'Glen Elm Park',
'Glen Elm Park South',
'Grand Trunk Annex',
'Harbour Landing',
'Highland Park',
'Hillsdale',
'Industrial Centre',
'Industrial Ross',
'Innismore',
'Kensington Greens',
'Lakeridge',
'Lakesidge Gardens',
'Lakeview',
'Lakeview Annex',
'Lakeview Centre',
'Lakeview South',
'Lakewood',
'Lewvan Park',
'Maple Ridge',
'Mayfair',
'McCarthy Park',
'Mirror Sun Gardens',
'Mount Pleasant',
'Mount Royal',
'Normandy Heights',
'Normanview',
'Normanview West',
'Normanview West Addition',
'North Regina',
'North Annex',
'Old 33',
'Parkridge',
'Parkview',
'Parliament Place',
'Pasqua Place',
'Premier Place',
'Regent Park',
'Richmond Place',
'River Bend',
'River Heights',
'Riverside',
'Rochdale Park',
'Rosemont',
'Rosemont North',
'Rosemont South',
'Rothwell Place',
'Sherwood Estates',
'South Lakeview see Lakeview South',
'Spruce Meadows',
'Transcona',
'Tuxedo Park',
'University Park',
'University Park East',
'Uplands',
'Varsity Park',
'Walsh Acres',
'Wascana Addition',
'Wascana Crescents',
'Wascana Park',
'Wascana View',
'Washington Park',
'Westhill',
'Westridge',
'Whitmore Park',
'Windsor Park',
'Windsor Place',
'Wood Meadows',
'Woodland Grove']
df['locations'] =0
df['locations'] = df['locations'].apply(lambda loc : listSubdivOfRegina[np.random.randint(0,len(listSubdivOfRegina))] )

df


# In[4]:


sel_Col=['age-range','gender','race','FoodChoice','field','reading_normalized','gaming_normalized','cleanliness_normalized','music_normalized','movies_normalized','tv_normalized','clubbing_normalized','hiking_normalized','exercise_normalized','sports_normalized','importance_same_religion_normalized','Smoking','Drinking','locations']
df=df[sel_Col]
field_preference_dict = {index: field for index, field in enumerate(df['field'].unique())}
df


# In[5]:


df.isnull().sum()


# In[6]:


df.dropna(inplace=True)
print("Rows with Null values are deleted\n\n")
duplicateRows = df[df.duplicated()]
duplicateRows


# In[7]:


duplicateRows.shape


# In[20]:


print("Total null value")
print(df.isnull().sum())
df = df.drop_duplicates()
print("Duplicate rows", df.duplicated().sum())
df


# In[9]:


df.to_csv('RCC_Cleaned.csv',index = False)


# In[10]:


new_df= pd.read_csv('RCC_Cleaned.csv')
new_df.describe()


# In[11]:


# Check for duplicates
print("duplicate rows:", new_df.duplicated().sum())


# In[12]:


new_df.shape


# In[13]:


c=1
for i in df:
    print(c,".", i, ": " ,df[i].dtype)
    c=c+1


# In[14]:


for column in new_df:
    if new_df[column].dtype == 'float64':
            icondition=new_df[(new_df[column] >= 0.1) & (new_df[column] <= 1.0)]
            new_df = icondition

# Print the DataFrame after removing outliers
print(new_df)
new_df.shape      


# In[15]:


# Filter outliers based on the specified range
lower_limit = 0.1
upper_limit = 1.0
for i in new_df:
        if new_df[i].dtypes == 'float64':
            

                outliers = new_df[(new_df[i] < lower_limit) | (new_df[i] > upper_limit)]

                # Create a scatter plot
                plt.figure(figsize=(12, 6))
                plt.scatter(df.index, df[i], color='blue', label='Data points')

                # Highlight outliers
                plt.scatter(outliers.index, outliers[i], color='red', label='Outliers')

                plt.axhline(y=lower_limit, color='green', linestyle='--', label='Lower Limit (0.1)')
                plt.axhline(y=upper_limit, color='orange', linestyle='--', label='Upper Limit (1.0)')
                
                plt.title(i)
                plt.xlabel('Population')
                plt.ylabel('Preference Range')
                plt.legend()
                plt.show()            


# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_categorical_bar(data, column, xlabel=None, ylabel='Count', title=None):
  
    os.makedirs('plots',exist_ok=True)
    plt.figure(figsize=(12,6))
    sns.countplot(data=data, x=column)
    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel(ylabel)
    plt.title(title if title else f'{column} Distribution')
    plt.xticks(rotation=45)  # Rotates x-axis labels for better readability
    plt.savefig(f"plots/{column}.png")
    plt.show()

# Example usage
for colName in list(new_df):
    plot_categorical_bar(new_df, column=colName, xlabel=colName, ylabel='Frequency', title=f'{colName} Distribution')


# In[17]:


# Count the occurrences of each category
category_counts = new_df['locations'].value_counts()

# Create a horizontal bar plot
plt.figure(figsize=(8, 20))
category_counts.plot(kind='barh', color='blue')

# Adding labels and title
plt.xlabel('Frequency')
plt.ylabel('Locations')
plt.title('Location Distribution')

# Display the plot
plt.show()


# In[18]:


# Count the occurrences of each category
category_counts = new_df['field'].value_counts()

# Create a horizontal bar plot
plt.figure(figsize=(10, 30))
category_counts.plot(kind='barh', color='blue')

# Adding labels and title
plt.xlabel('Frequency')
plt.ylabel('Field of Work')
plt.title('Field Distribution')

# Display the plot
plt.show()


# In[19]:


new_df.to_csv('RCC_Dataset.csv',index = False)
df1= pd.read_csv('RCC_Dataset.csv')
df1.shape


# In[ ]:




