import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import date
import os
import glob
import requests
import math
import random

warnings.simplefilter(action='ignore',category=FutureWarning)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def random_name(gen):
    # Lists of sample names
    first_names = ['Axy', 'Bmn', 'Cpq', 'Dij', 'Epo', 'Fst','Grs','Hbc']
    last_names =  ['101', '202', '303', '404', '505', '606','707','808']

    # Generate a random name
    random_first_name = random.choice(first_names)
    random_last_name = random.choice(last_names)
    random_full_name = f"{random_first_name} {random_last_name}"
    if gen == 'male':
        return "Mr. "+random_full_name
    else:
        return "Ms. "+random_full_name



def get_coordinates(api_key, city_name):
    base_url = "https://atlas.microsoft.com/search/address/json"
    params = {
        "api-version": "1.0",
        "subscription-key": api_key,
        "query": city_name
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if "results" in data and data["results"]:
        location = data["results"][0]["position"]
        latitude = location["lat"]
        longitude = location["lon"]
        return latitude, longitude
    else:
        return None, None
    
def get_numeric_input(prompt):
    selected_option_text = st.text_input(prompt)
    selected_option = None

    if selected_option_text:
        try:
            selected_option = float(selected_option_text)
            if 0 <= selected_option <= 1:
                st.write("You selected:", selected_option)
            else:
                st.write("Please enter a value between 0 and 1.")
        except ValueError:
            st.write("Please enter a valid numeric value.")
    
    return selected_option


def get_distance(from_city_name,to_city_name):
    
    api_key = "szf6w7lIXw-H0uNVASqmK2TqlaNO-LRXXbt91ODbEb8"
    from_latitude, from_longitude = get_coordinates(api_key, from_city_name)
    to_latitude, to_longitude = get_coordinates(api_key, to_city_name)
    distance = haversine_distance(from_latitude, from_longitude, to_latitude, to_longitude)
    # print(f"Distance between the cities: {distance:.2f} km")
    
    return np.round(distance,2)


def calculate_compatibility(user_preferences):
    # Reset compatibility score
    df['compatbilityScore'] = 0 
    df['compatbilityScorePercentage'] = 0 

    # Calculate compatibility score based on user preferences
    for index, row in df.iterrows():
        # print('Before Row : ',row)
        for key, value in user_preferences.items():
            if 'normalized' in key:
                dataset_value = row[key]
                external_value = value
                compact_score  = np.round(1 - np.abs(dataset_value - external_value),2)
                # row['compatbilityScore']+=compact_score
                df.at[index,'compatbilityScore'] +=compact_score
            else:                 
                if key=='age-range':
                    # print('key : ',key)
                    df.at[index,'compatbilityScore']+=2
                elif key== 'FoodChoice':
                    # print('key : ',key)
                    df.at[index,'compatbilityScore']+=2
                elif key=='Smoking':
                    # print('key : ',key)
                    df.at[index,'compatbilityScore']+=2
                elif key=='Drinking':
                    # print('key : ',key)
                    df.at[index,'compatbilityScore']+=2
                elif key=='race':
                    # print('key : ',key)

                    df.at[index,'compatbilityScore']+=2
                elif key=='gender':
                    # print('key : ',key)

                    df.at[index,'compatbilityScore']+=2
                
                elif key=='field':
                    # print('key : ',key)

                    df.at[index,'compatbilityScore']+=2
            
                else:
                    # print('key : ',key)

                    df.at[index,'compatbilityScore']+=1


    # Calculate the maximum compatibility score in the dataset
    max_compatibility_score = len(list(user_preferences.keys()))+7

    # Iterate through the DataFrame and calculate the normalized compatibility score as a percentage
    for index, row in df.iterrows():
        df.at[index,'compatbilityScorePercentage'] = np.round(int((row['compatbilityScore'] / max_compatibility_score) * 100), 2)
    
    compatible_df = df[df['compatbilityScorePercentage'] != 0]
    compatible_df = df.drop_duplicates()
    compatible_df.sort_values(by='compatbilityScorePercentage', ascending=False,inplace=True)
    return compatible_df

df = pd.read_csv('RCC_Dataset.csv')

st.set_page_config(layout='wide',page_icon='!',page_title='Roommate Compatibility Connection')
path =  os.path.dirname(__file__)
today = date.today()
# Column 1
# Title of the app
st.title("Roommate Compatibility Connection")
st.markdown("Use the options below to select your preferences for finding compatible roommates.")

# Create two columns
col1, col2,col3 = st.columns(3)


with col1:
    
    # Add style to differentiate columns
    st.markdown("<style>div[role='listbox'] { background-color: #f2f2f2; padding: 5px; }</style>", unsafe_allow_html=True)
    
    # Dropdown menu
    
    selected_option_age = st.selectbox("Select an option for Age :", sorted(df['age-range'].unique().tolist()))
    # Display the selected option
    st.write("You selected:", selected_option_age)

    selected_option_gender = st.selectbox("Select an option for Gender :", df['gender'].unique().tolist())
    # Display the selected option
    st.write("You selected:", selected_option_gender)

    selected_option_race = st.selectbox("Select an option for Ethnicity :", sorted(df['race'].unique().tolist()))
    # Display the selected option
    st.write("You selected:", selected_option_race)

    selected_option_field = st.selectbox("Select an option for Field of Work :", sorted(df['field'].unique().tolist()))
    # Display the selected option
    st.write("You selected:", selected_option_field)

    selected_option_food = st.selectbox("Select an option for Food Habit :", df['FoodChoice'].unique().tolist())
    # Display the selected option
    st.write("You selected:", selected_option_food)
    
    selected_option_smooking = st.selectbox("Select an option for Smooking Habit :", df['Smoking'].unique().tolist())
    # Display the selected option
    st.write("You selected:", selected_option_smooking)
    
    selected_option_drinking = st.selectbox("Select an option for Alcohol Consumption :", df['Drinking'].unique().tolist())
    # Display the selected option   
    st.write("You selected:", selected_option_drinking)
    
with col2:
    
    st.markdown("<style>div[role='listbox'] { background-color: #f2f2f2; padding: 5px; }</style>", unsafe_allow_html=True)
    selected_option_reading = get_numeric_input("Select an option for Academic (Preference order(0-1)):")  
    selected_option_cleanliness = get_numeric_input("Select an option for cleanliness and hygiene (Preference order(0-1)):") 
    selected_option_exercise = get_numeric_input("Select an option for Exercise and yoga(Preference order(0-1):") 
    selected_option_hiking = get_numeric_input("Select an option for Trip & Hiking(Preference order(0-1)):")
    selected_option_religion = get_numeric_input("Select an option for Religion Faith & Belief (Preference order(0-1)):")
    selected_option_movie = get_numeric_input("Select an option for Movie lover(Preference order(0-1)):")  
    selected_option_music = get_numeric_input("Select an option for Music Lover(Preference order(0-1)):") 
    selected_option_sports = get_numeric_input("Select an option for Sports Enthusiasm(Preference order(0-1)):") 
    selected_option_tv = get_numeric_input("Select an option for Binge-Watching (Preference order(0-1)):")    
    selected_option_gaming = get_numeric_input("Select an option for Gaming Habit(Preference order(0-1)):")   
    selected_option_clubbing = get_numeric_input("Select an option for party & Clubbing(Preference order(0-1)):")     
selected_option_origon_location = st.text_input("Enter Your University/office/Home location:")
# Display the entered text
st.write("You entered:", selected_option_origon_location)

if selected_option_origon_location:
    
    if st.button('Calculate Compatibility'):

        user_preferences = {'age-range': selected_option_age,
                    'gender': selected_option_gender,
                    'race': selected_option_race,
                    'field': selected_option_field,
                    'Smoking':selected_option_smooking,
                    'Drinking':selected_option_drinking,
                    'reading_normalized':selected_option_reading,
                    'gaming_normalized':selected_option_gaming,
                    'music_normalized':selected_option_music,
                    'cleanliness_normalized':selected_option_cleanliness,
                    'movies_normalized':selected_option_movie,
                    'tv_normalized':selected_option_tv,
                    'clubbing_normalized':selected_option_clubbing,
                    'hiking_normalized':selected_option_hiking,
                    'exercise_normalized':selected_option_exercise,
                    'sports_normalized':selected_option_sports,
                    'importance_same_religion_normalized':selected_option_religion,
                    'FoodChoice':selected_option_food

                }
        # print(user_preferences)
        # st.dataframe( pd.DataFrame.from_dict(user_preferences))
        compatibility_df = calculate_compatibility(user_preferences)
        # st.dataframe(compatibility_df)            
      
        st.title("Available Roommates - Sorted by Compatibility Score")
        col1,col2 = st.columns(2)

        with col1:

            compatible_sorted_df = compatibility_df.head()
            # st.dataframe(compatible_sorted_df) 
            # st.dataframe(compatible_sorted_df)            
            i=0
            for index,row in compatible_sorted_df.iterrows():
                distance = get_distance(row['locations'], selected_option_origon_location)
                i=i+1
                if distance:
                    st.markdown(
                        f"""
                        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 15px; box-shadow: 2px 2px 5px #888888;">
                        <h4>OPTION: {(i)} </h4>
                        <table border="1">
                            <tr>
                                <td><strong>Age Range:</strong> {row['age-range']}</td>
                                <td><strong>Gender:</strong> {row['gender']}</td>
                                <td><strong>Ethnicity:</strong> {row['race']}</td>
                            </tr>
                            <tr>
                                <td><strong>Field of Interest:</strong> {row['field']}</td>
                                <td><strong>Academic and Reading Preference:</strong> {row['reading_normalized']}</td>
                                <td><strong>Cleanliness and hygiene :</strong> {row['cleanliness_normalized']}</td>
                            </tr>
                            <tr>
                                <td><strong>Music Lover :</strong> {row['music_normalized']}</td>
                                <td><strong>Gaming Habit :</strong> {row['gaming_normalized']}</td>
                                <td><strong>Alcohol Consumption :</strong> {row['Drinking']}</td>  
                            </tr>
                            <tr>
                                <td><strong>Movie Lover:</strong> {row['movies_normalized']}</td>
                                <td><strong>Binge-Watching:</strong> {row['tv_normalized']}</td>
                                <td><strong>Party and Clubbing :</strong> {row['clubbing_normalized']}</td>
                            </tr>
                            <tr>
                                <td><strong>Trip & Hiking :</strong> {row['hiking_normalized']}</td>
                                <td><strong>Exercise and Yoga :</strong> {row['exercise_normalized']}</td>
                                <td><strong>Sports Enthusiasm :</strong> {row['sports_normalized']}</td>
                            </tr>
                            <tr>
                                <td><strong>Religion Faith & belief :</strong> {row['importance_same_religion_normalized']}</td>
                                <td><strong>Food Habit :</strong> {row['FoodChoice']}</td> 
                                <td><strong>Smooking Habit :</strong> {row['Smoking']}</td>
                            </tr>
                            <tr>
                                 <td><strong>Location :</strong> {row['locations']}</td>
                            </tr> 
                            
                            
                        </table>
                            
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        with col2:
            j=0
            compatible_sorted_df = compatibility_df.head()
            for index,row in compatible_sorted_df.iterrows():
                distance = get_distance(row['locations'], selected_option_origon_location)
                j=j+1
                if distance:
                    st.markdown(
                        f"""
                        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 15px; box-shadow: 2px 2px 5px #888888;">
                        <h3>OPTION: {(j)} </h3>
                        <h4><strong>Compatibility Score : </strong> {row['compatbilityScorePercentage']}% </h4>
                        <h5>Name: {random_name(row['gender'])} </h5>
                        <h5><strong>Location :</strong> {row['locations']}</h5>
                        <h5>Distance between : {row['locations']} - {selected_option_origon_location} : {distance/1000}-KM</h5>
                        
                        

                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                   
 
