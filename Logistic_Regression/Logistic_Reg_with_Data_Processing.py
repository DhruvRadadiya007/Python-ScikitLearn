import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

df = pd.read_csv("Logistic_Regression\Customer_Behaviour.csv")

print(df.head())

print(df.info())
print(df.describe())
print(df.shape)
print(df.nunique())

new_data = df.drop("User ID",axis=1)
print(new_data.head())

encoded_data = pd.get_dummies(new_data,columns=["Gender"],drop_first=True)

print(encoded_data.head())

x = encoded_data.drop("Purchased",axis=1)
y = encoded_data["Purchased"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = LogisticRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(y_pred)
print(model.score(x_test,y_test))

print(x_test.head())


# Predict the results for test data
y_pred = model.predict(x_test)

