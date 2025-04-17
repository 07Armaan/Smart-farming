import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv("smart_farming.csv")

x = df[['Soil_Moisture', 'Temperature', 'Rainfall', 'Humidity', 'pH_Level', 'Organic_Matter', 'Soil_Compaction']]  # Features
y = df['Water_Required'] 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

regression_std = StandardScaler()
x_train = regression_std.fit_transform(x_train)
x_test = regression_std.transform(x_test)

regression_pca = PCA(n_components=6)
x_train = regression_pca.fit_transform(x_train)
x_test = regression_pca.transform(x_test)

regression_model = RandomForestRegressor(random_state=42)
regression_model.fit(x_train,y_train)

y_pred = regression_model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(mse,r2)

joblib.dump((regression_model,regression_std,regression_pca),"regression.pkl")
print(x.head())