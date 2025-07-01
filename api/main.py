# api/main.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Mendefinisikan struktur input data menggunakan Pydantic
class StockFeatures(BaseModel):
    SMA_5: float
    Price_Change: float
    Volume: float

# Inisialisasi aplikasi FastAPI
app = FastAPI(title="Stock Predictor API", version="1.0")

# Muat model dari file lokal hasil training.
# Kita gunakan joblib untuk memuat model Random Forest yang telah disimpan.
model = joblib.load("model_local/random_forest_model.pkl")

@app.get("/", tags=["Health Check"])
async def root():
    """Endpoint untuk cek status API."""
    return {"message": "API is running!"}

@app.post("/predict", tags=["Prediction"])
async def predict(features: StockFeatures):
    """Endpoint untuk membuat prediksi harga saham (Naik/Turun)."""
    # Ubah input menjadi DataFrame yang bisa dibaca model
    feature_df = pd.DataFrame([features.dict()])

    # Buat prediksi
    prediction = model.predict(feature_df)

    # Ubah hasil (0 atau 1) menjadi label yang lebih mudah dibaca
    result = "Naik" if prediction[0] == 1 else "Turun"

    return {"prediction": result}

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}