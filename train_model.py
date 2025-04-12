# train_model.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import joblib

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Skip if model already exists
if os.path.exists("models/model_xgb.pkl"):
    print("✅ Model already trained.")
    exit()

np.random.seed(42)
customer_ids = [f"C{str(i+1).zfill(3)}" for i in range(50)]
df_coords = pd.DataFrame({
    "Customer ID": customer_ids,
    "Home Latitude": np.round(np.random.uniform(12.8, 13.0, 50), 4),
    "Home Longitude": np.round(np.random.uniform(77.5, 77.7, 50), 4),
    "Office Latitude": np.round(np.random.uniform(12.8, 13.0, 50), 4),
    "Office Longitude": np.round(np.random.uniform(77.5, 77.7, 50), 4),
})
df_coords.to_csv("data/customer_coordinates.csv", index=False)

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
                    "Is Weekend": int(trip_time.weekday() >= 5),
                    "Direction": direction,
                    "Start Latitude": slat,
                    "Start Longitude": slon,
                    "End Latitude": elat,
                    "End Longitude": elon,
                    "Lat Delta": abs(slat - elat),
                    "Lon Delta": abs(slon - elon),
                    "Distance (km)": round(distance_km, 2),
                    "ETA (min)": round(eta, 2),
                })

    return pd.DataFrame(trip_data)

trip_df = generate_trip_data(df_coords)
trip_df.to_csv("data/synthetic_trip_data.csv", index=False)

day_encoder = LabelEncoder().fit(trip_df["Day of Week"])
dir_encoder = LabelEncoder().fit(trip_df["Direction"])
trip_df["Day Encoded"] = day_encoder.transform(trip_df["Day of Week"])
trip_df["Direction Encoded"] = dir_encoder.transform(trip_df["Direction"])

X = trip_df[[
    "Start Latitude", "Start Longitude", "End Latitude", "End Longitude",
    "Distance (km)", "Day Encoded", "Direction Encoded", "Hour",
    "Is Weekend", "Lat Delta", "Lon Delta"
]]
y = trip_df["ETA (min)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "models/model_xgb.pkl")
joblib.dump(day_encoder, "models/day_encoder.pkl")
joblib.dump(dir_encoder, "models/direction_encoder.pkl")

print("✅ Model trained and saved.")
