import pandas as pd
import os
import imblearn
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

dir = "C:/Users/aiswariya.c/OneDrive - Infosys Limited/Finloop/Dataset_files"
file_path = os.path.join(dir, "indian_liver_patient.csv")
df = pd.read_csv(file_path, encoding = "unicode_escape")
print("data loaded")

# df["Gender"] = df["Gender"].astype("category")
df["Gender"] = df["Gender"].replace("Female",0) # 0 -female
df["Gender"] = df["Gender"].replace("Male",1) # 1 - male

# 0 -not suffering 1 - suffering with liver diesease
df["Result"] = df["Dataset"].replace(2,0)
df.drop("Dataset", axis = 1, inplace = True)

df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean())

print("data preprocessed completed")
X = df.drop('Result', axis=1)
y = df["Result"]
os = RandomOverSampler(random_state=42)
X_res,y_res = os.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
rfc = RandomForestClassifier(n_estimators = 200, bootstrap = True)
model = rfc.fit(X_train,y_train)
print("model trained")

with open('model.pkl' , 'wb') as file:
         pickle.dump(model,file)
         print("model saved")