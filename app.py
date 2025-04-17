from flask import Flask , render_template,request
import numpy as np
import joblib 

classification_model,classification_std,classification_pca,soil_Type_le_clas,nutrient_Content_le_clas,season_le_clas = joblib.load("classification_model.pkl")
regression_model,regression_std,regression_pca = joblib.load("regression.pkl")
cluster_model,cluster_std,cluster_pca,cluster_names = joblib.load("cluster_model.pkl")
recommendation_model,recommendation_std,recommendation_pca,soil_Type_le_recomm,season_le_recomm,Crop_Type_le_recomm,y = joblib.load("recommendation_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=["POST"])
def prediction():
    Soil_Type = request.form["Soil_Type"]
    Temperature = float(request.form["Temperature"])
    Rainfall = float(request.form["Rainfall"])
    Humidity = float(request.form["Humidity"])
    pH_Level = float(request.form["pH_Level"])
    Soil_Moisture = float(request.form["Soil_Moisture"])
    Organic_Matter = float(request.form["Organic_Matter"])
    Sunlight = float(request.form["Sunlight"])
    Nutrient_Content = request.form["Nutrient_Content"]
    Soil_Compaction = float(request.form["Soil_Compaction"])
    Season = request.form["Season"]
   
    Soil_Type_clas = soil_Type_le_clas.transform([Soil_Type])[0]
    nutrient_Content_clas = nutrient_Content_le_clas.transform([Nutrient_Content])[0]
    season_clas = season_le_clas.transform([Season])[0]

    class_test = np.array([[Soil_Type_clas,Temperature,Rainfall,Humidity,pH_Level,Soil_Moisture,Organic_Matter,Sunlight,nutrient_Content_clas,Soil_Compaction,season_clas]])

    class_test = classification_std.transform(class_test)
    class_test = classification_pca.transform(class_test)

    classification_pred = classification_model.predict(class_test)[0]
    classification_pred_name = "cannot clarify"

    if classification_pred==0:
        classification_pred_name = "NO"

    else :
        classification_pred_name = "YES"  
    #classification code ends
    
    #regression code starts
    regg_test = np.array([[Soil_Moisture,Temperature,Rainfall,Humidity,pH_Level,Organic_Matter,Soil_Compaction]])   
    regg_test = regression_std.transform(regg_test)
    regg_test = regression_pca.transform(regg_test)

    regg_pred = regression_model.predict(regg_test)[0] 
    #regression code ends here
    #cluster code starts
   
    cluster_test = np.array([[Soil_Moisture,Temperature,Rainfall,Humidity,pH_Level,Organic_Matter,Soil_Compaction]])
    cluster_test = cluster_std.transform(cluster_test)
    cluster_test = cluster_pca.transform(cluster_test)

    cluster_test = cluster_model.predict(cluster_test)
    cluster_name_predict = cluster_names.get(int(cluster_test))
    #the clustering code ends
   
    #recommendation code starts
    soil_Type_recomm = soil_Type_le_recomm.transform([Soil_Type])[0]
    season_recomm = season_le_recomm.transform([Season])[0]

    recomm_test = np.array([[soil_Type_recomm, Temperature, Rainfall, Soil_Moisture, pH_Level, Organic_Matter, season_recomm]])

    recomm_test = recommendation_std.transform(recomm_test)
    recomm_test = recommendation_pca.transform(recomm_test)

    recomm_distance,recomm_indices = recommendation_model.kneighbors(recomm_test)

    recomm_pred = y.iloc[recomm_indices.flatten()].values[0]
    recomm_name = Crop_Type_le_recomm.inverse_transform([recomm_pred])[0]


    return render_template("index.html",output =f"The classified prediction is {classification_pred_name},The regression prediction is {round(regg_pred)} liters,cluster prediction is {cluster_name_predict},the recommendation prediction is: {recomm_name}")

if __name__ == "__main__":
    import os
    app.run(debug=True,host="0.0.0.0",port=int(os.environ.get("PORT",5000)))
