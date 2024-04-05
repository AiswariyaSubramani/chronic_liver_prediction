from flask import Flask, render_template, request, redirect, url_for, request
import joblib
import json
import pandas as pd

app = Flask(__name__)

@app.route('/liver')
def liver():
    return render_template("liver.html")

@app.route('/predict') # , methods=['POST']
def predict():
    print("inside predict")
    to_predict = {"Age":53, "Gender":0,"Total_Bilirubin":2.8,"Direct_Bilirubin":12.5,"Alkaline_Phosphotase":150,
         "Alamine_Aminotransferase":50,"Aspartate_Aminotransferase":30, "Total_Protiens":5.5,"Total_Protiens":3.0,
        "Albumin":3,"Albumin_and_Globulin_Ratio":1.5}
    df_in = pd.DataFrame(to_predict, index = [0])
    model = joblib.load('model.pkl')
    result = model.predict(df_in)
    if(int(result)==1):
        prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return render_template("result.html", result_text = prediction)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)