import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#no encoded is used in training model

df = pd.read_csv("smart_farming.csv")

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

x = df[['Soil_Moisture', 'Temperature', 'Rainfall', 'Humidity', 'pH_Level', 'Organic_Matter', 'Soil_Compaction']]

cluster_std = StandardScaler()
x = cluster_std.fit_transform(x)

cluster_pca = PCA(n_components=0.1)
x = cluster_pca.fit_transform(x)

cluster_model = KMeans(n_clusters=3,random_state=42)
cluster_model.fit(x)

y = cluster_model.predict(x)

df["cluster"] = y

cluster_mean = df.groupby(y)[['Soil_Moisture', 'pH_Level', 'Organic_Matter']].mean()
print(cluster_mean)

cluster_names = {
    0: 'Wet Fertile Soil',
    1: 'Medium Quality Soil',
    2: 'Dry Soil'
}

df["cluster_names"] = df["cluster"].map(cluster_names)
print(df.head())

print(silhouette_score(x,y))

joblib.dump((cluster_model,cluster_std,cluster_pca,cluster_names),"cluster_model.pkl")
