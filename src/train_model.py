# Di dalam src/train_model.py

import pandas as pd
#import mlflow
#import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import joblib
import os

# ---- Kode Modifikasi ----
# Membaca data dari file yang dihasilkan oleh tahap get_data
data_path = 'data/raw/BBCA.JK.csv'
data = pd.read_csv(data_path)
# Hapus baris non-data (baris dengan 'Ticker' dan 'Date')
data = data[~data['Price'].isin(['Ticker', 'Date'])]
# Konversi kolom yang diperlukan ke tipe numerik
for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data['Date'] = pd.to_datetime(data['Price'])
data = data.set_index('Date')
# -----------------------

# Feature Engineering
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['Price_Change'] = data['Close'].diff()
data = data.dropna()

# Buat Target Variable
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

# Split data (time series)
split_index = int(len(data) * 0.8)
train = data.iloc[:split_index]
test = data.iloc[split_index:]

features = ['SMA_5', 'Price_Change', 'Volume']
X_train, y_train = train[features], train['Target']
X_test, y_test = test[features], test['Target']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict & evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Akurasi model di data test:", accuracy)

# Logging ke MLflow (opsional, jika sudah setup)
# with mlflow.start_run() as run:
#     mlflow.sklearn.log_model(model, "model")
#     mlflow.log_metric("accuracy", accuracy)
#     print(f"MLflow run_id: {run.info.run_id}")

# Pastikan direktori model_local ada
os.makedirs("model_local", exist_ok=True)
joblib.dump(model, "model_local/random_forest_model.pkl")
print("Model saved to model_local/random_forest_model.pkl")

# --- Animasi: Visualisasi prediksi naik/turun ---
fig, ax = plt.subplots(figsize=(10,5))
prices = test['Close'].values
preds = predictions

line, = ax.plot([], [], lw=2, label='Harga Close')
scat = ax.scatter([], [], c=[], cmap='bwr', label='Prediksi (Naik=1, Turun=0)')
ax.set_xlim(0, len(prices))
ax.set_ylim(prices.min()*0.98, prices.max()*1.02)
ax.set_title('Animasi Prediksi Naik/Turun')
ax.set_xlabel('Hari')
ax.set_ylabel('Harga')
ax.legend()

# Fungsi update animasi
def update(frame):
    if frame == 0:
        line.set_data([], [])
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array([])
        return line, scat
    line.set_data(range(frame), prices[:frame])
    offsets = np.column_stack((range(frame), prices[:frame]))
    scat.set_offsets(offsets)
    scat.set_array(preds[:frame])
    return line, scat

ani = animation.FuncAnimation(fig, update, frames=len(prices), interval=30, blit=True)
plt.show()