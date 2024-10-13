import pandas as pd
import numpy as np
import streamlit as st
import os
import pickle
import requests
import json
from streamlit.components.v1 import html
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

save_path = 'C://Users//berni//OneDrive//Desktop//FYP Coding'

os.makedirs(save_path, exist_ok=True)
encoding_path = os.path.join(save_path, 'encoded_cols.pkl')

with open(encoding_path, 'rb') as file:
    encoded_cols = pickle.load(file)
 

# Arrest Model
os.makedirs(save_path, exist_ok=True)
model_path = os.path.join(save_path, 'arrest.pkl')

with open(model_path, 'rb') as f:
    ada_clf = pickle.load(f)

# Category Model
os.makedirs(save_path, exist_ok=True)
model_path = os.path.join(save_path, 'category.pkl')

with open(model_path, 'rb') as f:
    catboost_model = pickle.load(f)

# Spatial Model
os.makedirs(save_path, exist_ok=True)
model_path = os.path.join(save_path, 'spatial.pkl')

with open(model_path, 'rb') as f:
    rf_regressor = pickle.load(f)

# Preprocessing functions
def preprocess_input(data):

    input_df = pd.DataFrame([data])

    # Convert boolean columns to integers
    input_df['Domestic'] = input_df['Domestic'].astype(int)

    # List of numerical and categorical columns 
    numerical_cols = ['Beat', 'District', 'Ward', 'Community Area', 'Day', 'Month', 'Hour']
    categorical_cols = ['Location Description','Day_of_Week', 'Time_of_Day']

    # Convert numerical columns to numeric types and handle missing values
    for col in numerical_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    # Encode categorical columns as numerical values
    for col in categorical_cols:
        # Ensure input_df[col] is iterable and not a single value
        input_df[col] = pd.Categorical(input_df[col], categories=input_df[col].unique()).codes
    
    return input_df


# GPT Recommendation Function
def get_recommendation(user_input, predictions):
    api_key = os.getenv("OPENAI_API_KEY")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = f"""
        Based on the following user input and predictions, what recommendations can you provide?\n\n
        User Input:\n{user_input}\n\n
        Predictions:\n{predictions}
    """
    data = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
 
    response = requests.post("https://api.wlai.vip/v1/chat/completions", headers=headers, json=data)

    try:
        response = requests.post("https://api.wlai.vip/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()  # Raise an error for bad status codes
        response_data = response.json()

        # Check if 'choices' key is in the response
        if 'choices' in response_data:
            recommendation = response_data["choices"][0]["message"]["content"]
        else:
            st.error(f"Unexpected API response structure: {json.dumps(response_data, indent=2)}")
            recommendation = "No recommendations available due to unexpected API response."

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
        st.error(f"Response content: {response.text if response else 'No response'}")
        recommendation = "Error in getting recommendations."
    
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON response: {str(e)}")
        st.error(f"Response content: {response.text if response else 'No response'}")
        recommendation = "Error in getting recommendations."

    return recommendation


def create_label_with_info(label_text, info_text=None, key=None, input_type="text", options=None, margin_top="10px"):
    """
    Creates a Markdown label with an optional info icon for various input types, with optional space above the label.

    Args:
        label_text (str): The text for the label.
        info_text (str): The text to show in the info icon tooltip (optional).
        key (str): The key for the Streamlit widget.
        input_type (str): The type of input widget ("text", "number", "selectbox"). Defaults to "text".
        options (list): The options for selectbox. Defaults to None.
        margin_top (str): The space above the Markdown label. Defaults to "10px".
    
    Returns:
        The value from the Streamlit widget.
    """
    # Define the Markdown with CSS and optional info icon
    if info_text:
        markdown = f"""
        <style>
            .input-label {{
                display: flex;
                align-items: center;
                margin-top: {margin_top}; /* Add space above the label */
                margin-bottom: 0; /* Reduce space below the label */
            }}
            .input-label label {{
                margin-right: 5px;
                font-weight: bold; /* Make the label text bold */
            }}
            .input-label span {{
                margin-left: 5px;
                color: grey; /* Change the color to grey */
                cursor: pointer;
            }}
        </style>
        <div class="input-label">
            <label for="{key}">{label_text}</label>
            <span title="{info_text}">i</span>
        </div>
        """
    else:
        markdown = f"""
        <style>
            .input-label {{
                display: flex;
                align-items: center;
                margin-top: {margin_top}; /* Add space above the label */
                margin-bottom: 0; /* Reduce space below the label */
            }}
            .input-label label {{
                margin-right: 5px;
                font-weight: bold; /* Make the label text bold */
            }}
        </style>
        <div class="input-label">
            <label for="{key}">{label_text}</label>
        </div>
        """

    st.markdown(markdown, unsafe_allow_html=True)
    
    # Return the appropriate input widget based on the type
    if input_type == "text":
        return st.text_input(label_text, key=key, label_visibility="hidden")
    elif input_type == "number":
        return st.number_input(label_text, key=key, label_visibility="hidden")
    elif input_type == "selectbox":
        return st.selectbox(label_text, options, key=key, label_visibility="hidden")
    else:
        raise ValueError(f"Unsupported input type: {input_type}")
    
    
# Main Function
def main():
    st.title("🚨 Crime Prediction System")
    
    # Usage examples
    location_description = create_label_with_info(
        "Location Description",
        "The description of the location where the crime occurred, e.g., 'Park', 'Street'.",
        "location_description",
        "text"
    )

    domestic = create_label_with_info(
        "Domestic",
        None,
        "domestic",
        "selectbox",
        [True, False]
    )

    beat = create_label_with_info(
        "Beat",
        "Smallest police geographic area with a dedicated patrol car.",
        "beat",
        "number"
    )

    district = create_label_with_info(
        "District",
        "Police district number.",
        "district",
        "number"
    )

    ward = create_label_with_info(
        "Ward",
        "City Council district where the incident occurred.",
        "ward",
        "number"
    )

    community_area = create_label_with_info(
        "Community Area",
        "Chicago's 77 community areas where the incident occurred.",
        "community_area",
        "number"
    )

    day = create_label_with_info(
        "Day",
        None,
        "day",
        "number"
    )

    month = create_label_with_info(
        "Month",
        None,
        "month",
        "number"
    )

    day_of_week = create_label_with_info(
        "Day of Week",
        None,
        "day_of_week",
        "selectbox",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    hour = create_label_with_info(
        "Hour",
        None,
        "hour",
        "number"
    )

    time_of_day = create_label_with_info(
        "Time of Day",
        None,
        "time_of_day",
        "selectbox",
        ["Morning", "Afternoon", "Evening", "Night"]
    )


    # Prepare input data for preprocessing
    input_data = {
        'Location Description': location_description,
        'Domestic': domestic,
        'Beat': beat,
        'District': district,
        'Ward': ward,
        'Community Area': community_area,
        'Day': day,
        'Month': month,
        'Day_of_Week': day_of_week,
        'Hour': hour,
        'Time_of_Day': time_of_day
    }
    
    if st.button("Predict"):
        # Preprocess the input data
        preprocessed_input = preprocess_input(input_data)
        
        # Model 1: Predict "Arrest"
        arrest = ada_clf.predict(preprocessed_input)[0]
        arrest_status = 'True' if arrest == 1 else 'False'
        st.write(f"Arrest Prediction: {arrest_status}")
        
        # Model 2: Predict "Category" 
        preprocessed_input['Arrest'] = arrest
        category_array = catboost_model.predict(preprocessed_input)
        category_index = category_array[0].item()                                                           # Directly extract the scalar value
        reverse_encoded_cols = {k: {v: k for k, v in values.items()} for k, values in encoded_cols.items()} # Prepare the reverse encoding dictionary
        encoded_category = reverse_encoded_cols['Category'].get(category_index, "Unknown")                  # Lookup the original category label
        st.write(f"Category Prediction: {encoded_category}")
        
        # Model 3: Predict "Latitude" and "Longitude" 
        preprocessed_input['Category'] = category_index
        lat_lon = rf_regressor.predict(preprocessed_input)[0]
        st.write(f"Latitude Prediction: {lat_lon[0]}")
        st.write(f"Longitude Prediction: {lat_lon[1]}")  
        st.markdown(
        "<p style='font-size: 12px; color: blue;'><i>The blue marker represents the predicted coordinates.</i></p>",
        unsafe_allow_html=True
        )
        # Initialize Nominatim geocoder
        geolocator = Nominatim(user_agent="crime_prediction_system")
        # Get location name using geocoding
        location = geolocator.reverse((lat_lon[0], lat_lon[1]), language='en')
        location_name = location.address if location else "Unknown Location"

        # Create a folium map centered at the predicted location
        m = folium.Map(location=[lat_lon[0], lat_lon[1]], zoom_start=15)

        # Add a marker with a label
        folium.Marker(
            location=[lat_lon[0], lat_lon[1]],
            popup=folium.Popup(location_name, parse_html=True)
        ).add_to(m)

        # Render the folium map in Streamlit
        map_html = m._repr_html_()
        html(map_html, height=500, width=700)
    

        # Get recommendations from ChatGPT
        user_input_str = f"Location Description: {location_description}, Domestic: {domestic}, Beat: {beat}, District: {district}, Ward: {ward}, Community Area: {community_area}, Day: {day}, Month: {month}, Day_of_Week: {day_of_week}, Hour: {hour}, Time_of_Day: {time_of_day}"
        predictions_str = f"Arrest Prediction: {arrest_status}, Category Prediction: {encoded_category}, Latitude Prediction: {lat_lon[0]}, Longitude Prediction: {lat_lon[1]}"
        recommendation = get_recommendation(user_input_str, predictions_str) 
        st.write(f"**Recommendation from ChatGPT:**  \n{recommendation}")
    

if __name__ == "__main__":
    main()


