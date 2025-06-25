# Import library yang dibutuhkan
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Ambil Data
ticker = 'BBCA.JK'
data = yf.download(ticker, start='2020-01-01', end='2025-06-24')

print("Data Awal:")
print(data.head())

# 2. Feature Engineering Sederhana
# Kita buat fitur sederhana: perubahan harga dari 5 hari sebelumnya
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['Price_Change'] = data['Close'].diff()
data = data.dropna()

# 3. Buat Target Variable (Klasifikasi: Naik atau Turun)
# Jika harga penutupan besok > hari ini, maka 1 (Naik), selain itu 0 (Turun)
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna() # Drop baris terakhir karena tidak punya target

print("\nData dengan Fitur dan Target:")
print(data.head())

# 4. Siapkan Data untuk Model
features = ['SMA_5', 'Price_Change', 'Volume']
X = data[features]
y = data['Target']

# PENTING: Untuk data time-series, jangan diacak saat split!
# Kita ambil 80% data awal untuk training, 20% akhir untuk testing.
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"\nUkuran data training: {len(X_train)} baris")
print(f"Ukuran data testing: {len(X_test)} baris")

# 5. Latih Model Sederhana
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluasi Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nAkurasi model di data test: {accuracy:.4f}")

# Akurasi sekitar 50% sangat umum, ini menunjukkan betapa sulitnya pasar diprediksi.