import pandas as pd 
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("datasets\Customer_Behaviour.csv") 
print(data.head())

encoded_data = pd.get_dummies(data=data,columns=["Gender"],drop_first=True)
print(encoded_data.head())

final_data = encoded_data.drop("User ID",axis=1)
print(final_data.head())

x = final_data.drop("Purchased",axis=1)
y = final_data["Purchased"]

model = DecisionTreeClassifier(criterion="entropy")

model.fit(x,y)

new_customer = pd.DataFrame({
    
    "Gender" : ["Male"],
    "Age" : ["22"],
    "EstimatedSalary" : ["25000"],
    
})

encoded_customer = pd.get_dummies(data=new_customer,columns=["Gender"])
print(encoded_customer.head())
prediction = model.predict(encoded_customer)
print(prediction)
print(model.score(x,y))
