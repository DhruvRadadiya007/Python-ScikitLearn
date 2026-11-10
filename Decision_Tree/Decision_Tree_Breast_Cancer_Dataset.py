import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
x = data.data
y= data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.25)

model = DecisionTreeClassifier(criterion="entropy")

model.fit(x_train,y_train)

print(model.score(x_test,y_test))