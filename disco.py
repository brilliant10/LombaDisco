import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_disco.csv')
df.sample(5)

df.shape

df.info()

print("=== Jumlah Missing Value ===")
for i in df.columns:
    print(i,":",df[i].isna().sum())

df = df.dropna()
df = df.drop(columns=["id_umkm"])

numeric_cols = ["tenaga_kerja_perempuan", "tenaga_kerja_laki_laki", "aset",
                      "omset", "kapasitas_produksi", "laba", "biaya_karyawan", "jumlah_pelanggan"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
df[numeric_cols] = df[numeric_cols].fillna(0)

df = df.astype({col: 'int64' for col in numeric_cols})

df.info()

# Check the unique values in each categorical columns
for kategorikal in df.columns[df.dtypes == 'object']:
    print(kategorikal," :",df[kategorikal].nunique())
    print(df[kategorikal].unique())
    print("\n=========================================")

correlation_matrix = df.select_dtypes('number').corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title("Heatmap")
plt.show()

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

X = df.drop(columns=['nama_usaha', 'jumlah_pelanggan'], errors='ignore')
y = df['jumlah_pelanggan']

numeric_features = X.select_dtypes(include=['number']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Regression pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Hyperparameter grid
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [10, 20, 30, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

from sklearn.cluster import KMeans
X = df.drop(columns=['nama_usaha'], errors='ignore')
# Identifikasi fitur numerik dan kategorikal
numeric_features = X.select_dtypes(include=['number']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('clustering', KMeans(n_clusters=3, random_state=42))
])

pipeline.fit(X)

# Ambil label cluster
cluster_labels = pipeline.named_steps['clustering'].labels_

# Tambahkan ke dataframe
df['cluster'] = cluster_labels

df.groupby('cluster').mean(numeric_only=True)

import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline

inertia = []
k_range = range(1, 10)
for k in k_range:
    model = make_pipeline(preprocessor, KMeans(n_clusters=k, random_state=42))
    model.fit(X)
    inertia.append(model.named_steps['kmeans'].inertia_)

plt.plot(k_range, inertia, marker='o')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Contoh: status legalitas per cluster
print(df.groupby(['cluster', 'status_legalitas']).size())

# Contoh: marketplace per cluster
print(df.groupby(['cluster', 'marketplace']).size())

cluster_label_mapping = {
    0: 'UMKM Tumbuh',
    1: 'UMKM Tradisional',
    2: 'UMKM Potensial'
}

df['jenis_UMKM'] = df['cluster'].map(cluster_label_mapping)

# Fitur sebagai input
X = df.drop(columns=['nama_usaha', 'cluster', 'jenis_UMKM'], errors='ignore')

# Target klasifikasi adalah label cluster
y = df['jenis_UMKM']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Pisahkan kolom numerik dan kategorikal
numeric_features = X.select_dtypes(include='number').columns
categorical_features = X.select_dtypes(include='object').columns

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 🚨 Pipeline baru untuk klasifikasi — pastikan ini yang digunakan!
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Parameter grid untuk RandomForestClassifier
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__max_features': ['sqrt']
}

# Grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Evaluasi
print("Best Parameters:", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC AUC Score: {roc_auc:.4f}')
