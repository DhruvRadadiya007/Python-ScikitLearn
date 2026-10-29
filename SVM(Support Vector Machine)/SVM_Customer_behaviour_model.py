import pandas as pd 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

df = pd.read_csv("Logistic_Regression\Customer_Behaviour.csv")

print(df.head())

new_data = df.drop("User ID",axis=1)
print(new_data.head())

encoded_data = pd.get_dummies(new_data,"Gender",drop_first=True)
print(encoded_data.head())

x = encoded_data.drop("Purchased",axis=1)
y = encoded_data["Purchased"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = SVC(kernel="linear",C=1)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(model.score(x_test,y_test))

