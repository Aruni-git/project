import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report,ConfusionMatrixDisplay

data = pd.read_csv(r'/Users/aruni/VScode_Workspace/project/dataset/data.csv')
data.head()
data.describe()
data.isnull().sum()
data.duplicated().sum()
data.drop(['Unnamed: 32','id'],axis=1,inplace=True)
y = data['diagnosis']
X = data.drop(['diagnosis'],axis=1)
y.unique()
y.value_counts()
y = y.map({'M':1,'B':0})
y.value_counts()
X.head()
def high_corr(data,threshold):
    highly_corr = []
    for i in range(len(data.corr().columns)):
        for j in range(i):
            if abs(data.corr().iloc[i,j]) > threshold:
                print(f'({data.corr().columns[i]},{data.corr().columns[j]}) : {data.corr().iloc[i,j]}')
                highly_corr.append((data.corr().columns[i],data.corr().columns[j]))
    return highly_corr

high_corr_list = high_corr(X,0.95)
high_corr_list
X.drop(['perimeter_mean','area_mean','perimeter_se','area_se','perimeter_worst','area_worst'],axis=1,inplace=True)
X.drop(['fractal_dimension_mean','texture_se','smoothness_se','symmetry_se','fractal_dimension_se',],axis=1,inplace=True)
X = StandardScaler().fit_transform(X)
X_,X_test,Y_,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train,X_val,y_train,y_val = train_test_split(X_,Y_,test_size=0.3,random_state=42)
print(X_train.shape,X_val.shape,X_test.shape)
# Scale the features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Initialize the MLPClassifier with more iterations and enable early stopping
model = MLPClassifier(max_iter=1000, early_stopping=True)

# Fit the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Make predictions on the scaled validation and test sets
y_pred_val = model.predict(X_val_scaled)
y_pred_test = model.predict(X_test_scaled)

# Print accuracy scores for the validation and test sets
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val)}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test)}")
# Making predictions
predictions = model.predict(X_test)

# Now you can use classification_report
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print(classification_report(y_test, predictions))

import tensorflow as tf
from keras.models import load_model

# Assume 'model' is your MLPClassifier object
joblib.dump(model, 'ann.h5')


