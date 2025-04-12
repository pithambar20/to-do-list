# eta_predictor.py

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import pydeck as pdk
import os
import xgboost as xgb
import requests
import polyline

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Step 1: Generate synthetic coordinates for 50 customers
np.random.seed(42)
customer_ids = [f"C{str(i+1).zfill(3)}" for i in range(50)]
df_coords = pd.DataFrame({
    "Customer ID": customer_ids,
    "Home Latitude": np.random.uniform(12.8, 13.0, 50),
    "Home Longitude": np.random.uniform(77.5, 77.7, 50),
    "Office Latitude": np.random.uniform(12.8, 13.0, 50),
    "Office Longitude": np.random.uniform(77.5, 77.7, 50),
})
df_coords.to_csv("data/customer_coordinates.csv", index=False)

# Step 2: Generate synthetic trip data
def generate_trip_data(df, num_days=90):
    trip_data = []
    time_slots = {"morning": (7, 10), "evening": (17, 20)}
    start_date = datetime(2023, 1, 1)

    for i in range(num_days):
        day = start_date + timedelta(days=i)
        for _, row in df.iterrows():
            for trip_type in ["morning", "evening"]:
                hour = np.random.randint(*time_slots[trip_type])
                minute = np.random.randint(0, 60)
                trip_time = datetime.combine(day, datetime.min.time()) + timedelta(hours=hour, minutes=minute)
                direction = "Home_to_Office" if trip_type == "morning" else "Office_to_Home"
                slat, slon = (row["Home Latitude"], row["Home Longitude"]) if direction == "Home_to_Office" else (row["Office Latitude"], row["Office Longitude"])
                elat, elon = (row["Office Latitude"], row["Office Longitude"]) if direction == "Home_to_Office" else (row["Home Latitude"], row["Home Longitude"])

                # Haversine distance
                R = 6371
                lat1, lon1, lat2, lon2 = map(np.radians, [slat, slon, elat, elon])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                distance_km = R * c

                speed = max(10, (25 if trip_type == "morning" else 22) + np.random.normal(0, 2))
                eta = (distance_km / speed) * 60

                trip_data.append({
                    "Customer ID": row["Customer ID"],
                    "Date": trip_time.date(),
                    "Time": trip_time.time(),
                    "Hour": trip_time.hour,
                    "Day of Week": trip_time.strftime("%A"),
                    "Direction": direction,
                    "Start Latitude": slat,
                    "Start Longitude": slon,
                    "End Latitude": elat,
                    "End Longitude": elon,
                    "Distance (km)": round(distance_km, 2),
                    "ETA (min)": round(eta, 2),
                })

    return pd.DataFrame(trip_data)

trip_df = generate_trip_data(df_coords)
trip_df.to_csv("data/synthetic_trip_data.csv", index=False)

# Step 3: Train XGBoost model
day_encoder = LabelEncoder().fit(trip_df["Day of Week"])
dir_encoder = LabelEncoder().fit(trip_df["Direction"])
trip_df["Day Encoded"] = day_encoder.transform(trip_df["Day of Week"])
trip_df["Direction Encoded"] = dir_encoder.transform(trip_df["Direction"])

X = trip_df[[
    "Start Latitude", "Start Longitude", "End Latitude", "End Longitude",
    "Distance (km)", "Day Encoded", "Direction Encoded", "Hour"
]]
y = trip_df["ETA (min)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    objective='reg:squarederror',
    tree_method='auto',
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ Model trained! MAE: {mae:.2f} min, R²: {r2:.4f}")

joblib.dump(model, "models/model_xgb.pkl")
joblib.dump(day_encoder, "models/day_encoder.pkl")
joblib.dump(dir_encoder, "models/direction_encoder.pkl")

# Step 4: Streamlit App
st.set_page_config(page_title="ETA Predictor", layout="centered")

st.title("🚗 ETA Predictor with Route Map (OpenStreetMap)")

model = joblib.load("models/model_xgb.pkl")
day_encoder = joblib.load("models/day_encoder.pkl")
dir_encoder = joblib.load("models/direction_encoder.pkl")

with st.form("eta_form"):
    st.subheader("Enter Trip Details")
    start_lat = st.number_input("Start Latitude", value=12.9716)
    start_lon = st.number_input("Start Longitude", value=77.5946)
    end_lat = st.number_input("End Latitude", value=12.9352)
    end_lon = st.number_input("End Longitude", value=77.6146)
    hour = st.slider("Hour of Trip (24h)", 0, 23, 8)
    day = st.selectbox("Day of the Week", list(day_encoder.classes_))
    direction = st.selectbox("Direction", list(dir_encoder.classes_))
    submit = st.form_submit_button("Predict ETA")

if submit:
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    distance = haversine(start_lat, start_lon, end_lat, end_lon)
    day_encoded = day_encoder.transform([day])[0]
    direction_encoded = dir_encoder.transform([direction])[0]

    input_data = [[start_lat, start_lon, end_lat, end_lon, distance, day_encoded, direction_encoded, hour]]
    eta = model.predict(input_data)[0]

    st.info(f"📏 **Distance**: `{distance:.2f} km`")
    st.success(f"⏱️ **Estimated ETA**: `{eta:.2f} minutes`")

    # Request route from OSRM (OpenStreetMap)
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=polyline"
    try:
        response = requests.get(osrm_url)
        response.raise_for_status()
        route_data = response.json()
        route_coords = polyline.decode(route_data["routes"][0]["geometry"])

        # Create segment pairs
        segments = []
        for i in range(len(route_coords) - 1):
            segments.append({
                "from_lon": route_coords[i][1],
                "from_lat": route_coords[i][0],
                "to_lon": route_coords[i + 1][1],
                "to_lat": route_coords[i + 1][0]
            })
        segments_df = pd.DataFrame(segments)

        # Map visualization
        st.subheader("🗺️ Routed Path on Map")
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=np.mean([start_lat, end_lat]),
                longitude=np.mean([start_lon, end_lon]),
                zoom=12,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "LineLayer",
                    data=segments_df,
                    get_source_position='[from_lon, from_lat]',
                    get_target_position='[to_lon, to_lat]',
                    get_color=[255, 0, 0],
                    get_width=5,
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    data=pd.DataFrame([
                        {"lat": start_lat, "lon": start_lon},
                        {"lat": end_lat, "lon": end_lon}
                    ]),
                    get_position='[lon, lat]',
                    get_color='[0, 0, 255]',
                    get_radius=100,
                )
            ]
        ))
    except Exception as e:
        st.error(f"Routing failed: {e}")
