# src/get_data.py
import yfinance as yf
import pandas as pd
import os

# ---- Kode Baru ----
# Definisikan path output
output_dir = 'data/raw'
output_file = os.path.join(output_dir, 'BBCA.JK.csv')

# Buat direktori jika belum ada
os.makedirs(output_dir, exist_ok=True)
# ------------------

# Ambil Data
ticker = 'BBCA.JK'
data = yf.download(ticker, start='2020-01-01', end='2025-06-24')


# ---- Kode Baru ----
# Simpan data ke file CSV
data.to_csv(output_file, index_label='Date')
# ------------------

print(f"Data berhasil diunduh dan disimpan di {output_file}")