import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df = pd.read_csv("smart_farming.csv")

soil_Type_le_clas = LabelEncoder()
df["Soil_Type"] = soil_Type_le_clas.fit_transform(df["Soil_Type"])

nutrient_Content_le_clas = LabelEncoder()
df["Nutrient_Content"] = nutrient_Content_le_clas.fit_transform(df["Nutrient_Content"])

season_le_clas = LabelEncoder()
df["Season"] = season_le_clas.fit_transform(df["Season"])

x = df[[
    'Soil_Type', 'Temperature', 'Rainfall', 'Humidity','pH_Level', 'Soil_Moisture', 'Organic_Matter','Sunlight', 'Nutrient_Content', 'Soil_Compaction', 'Season']
]
y = df['Is_Suitable_For_Agriculture']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

classification_std = StandardScaler()
x_train = classification_std.fit_transform(x_train)
x_test = classification_std.transform(x_test)

classification_pca = PCA(n_components=4)
x_train = classification_pca.fit_transform(x_train)
x_test = classification_pca.transform(x_test)

classification_model = RandomForestClassifier(n_estimators=7,random_state=42)
classification_model.fit(x_train,y_train)

y_pred = classification_model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
cr = classification_report(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)

print(acc,cr,cm)

joblib.dump((classification_model,classification_std,classification_pca,soil_Type_le_clas,nutrient_Content_le_clas,season_le_clas),"classification_model.pkl")

print(season_le_clas.classes_)