import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df = pd.read_csv("smart_farming.csv")

soil_Type_le_recomm = LabelEncoder()
df["Soil_Type"] = soil_Type_le_recomm.fit_transform(df["Soil_Type"])

season_le_recomm = LabelEncoder()
df["Season"] = season_le_recomm.fit_transform(df["Season"])

Crop_Type_le_recomm = LabelEncoder()
df["Crop_Type"] = Crop_Type_le_recomm.fit_transform(df["Crop_Type"])


x =  df[['Soil_Type', 'Temperature', 'Rainfall', 'Soil_Moisture', 'pH_Level', 'Organic_Matter', 'Season']]
y = df['Crop_Type']  



#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
x_test = np.array([[2, 28.5, 1100, 60.0, 6.7, 4.2, 1]])
recommendation_std = StandardScaler()
x = recommendation_std.fit_transform(x)
x_test = recommendation_std.transform(x_test)

recommendation_pca = PCA(n_components=4)
x = recommendation_pca.fit_transform(x)
x_test = recommendation_pca.transform(x_test)

recommendation_model = NearestNeighbors(n_neighbors=3)
recommendation_model.fit(x)

distances,indices = recommendation_model.kneighbors(x_test)

recommendation = y.iloc[indices.flatten()].values[0]
recom_name = Crop_Type_le_recomm.inverse_transform([recommendation])[0]
print(recommendation)
print(recom_name)

joblib.dump((recommendation_model,recommendation_std,recommendation_pca,soil_Type_le_recomm,season_le_recomm,Crop_Type_le_recomm,y),"recommendation_model.pkl")