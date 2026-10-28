import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = pd.read_csv("Logistic_Regression\diabetes2.csv")

print(data.info())
print(data.describe())

cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
data[cols] = data[cols].replace(0,np.nan)
print(data.head())
data = data.dropna()
print(data.head())

x = data.drop("Outcome",axis=1)
y = data[["Outcome"]]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LogisticRegression()

model.fit(x_train,y_train)

print(model.score(x_test,y_test))

pred = model.predict(x_test)

cm = confusion_matrix(y_test,pred)

dis = ConfusionMatrixDisplay(confusion_matrix= cm, display_labels=['Positive','Negative'])
dis.plot()
plt.show()


