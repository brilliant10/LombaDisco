
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load data
df = pd.read_csv('dataset_disco.csv')

# Drop NA dan kolom ID
if "id_umkm" in df.columns:
    df = df.drop(columns=["id_umkm"])
df = df.dropna()

# Pastikan kolom numerik jadi int
numeric_cols = [
    "tenaga_kerja_perempuan", "tenaga_kerja_laki_laki", "aset",
    "omset", "kapasitas_produksi", "laba", "biaya_karyawan", "jumlah_pelanggan"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
df[numeric_cols] = df[numeric_cols].fillna(0)
df = df.astype({col: 'int64' for col in numeric_cols if col in df.columns})

# Ganti kolom berikut sesuai nama target di dataset!
target_col = 'target'  # <-- UBAH JIKA BERBEDA!
if target_col not in df.columns:
    raise Exception(f"Kolom target '{target_col}' tidak ditemukan. Ganti nama kolom target pada script ini!")

X = df.drop(columns=[target_col])
y = df[target_col]

# Cek dan encode kategorikal jika ada
categorical_cols = X.select_dtypes(include='object').columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict dan evaluasi
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC AUC Score: {roc_auc:.4f}')
