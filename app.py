from fastapi import FastAPI
import joblib

app = FastAPI()

# Load the trained model and scaler
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.post("/predict")
def predict(
    area_sqft: float,
    bedrooms: int,
    bathrooms: int,
    age_of_house: float,
    distance_to_city: float,
    parking: int,
    floor: int,
    nearby_schools: int,
    crime_rate: float
):
    new_house = [[
        area_sqft,
        bedrooms,
        bathrooms,
        age_of_house,
        distance_to_city,
        parking,
        floor,
        nearby_schools,
        crime_rate
    ]]

    new_house_scaled = scaler.transform(new_house)

    prediction = model.predict(new_house_scaled)

    return {
        "predicted_house_price": float(prediction[0])
    }