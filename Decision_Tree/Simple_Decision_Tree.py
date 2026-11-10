import pandas as pd 
from sklearn.tree import DecisionTreeClassifier

data = {
    
"study_hours": [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8],
"attendance": [80,90,75,90,87,90,35,60,90,98,90,80,90,80,75,90],
"result": ["Fail","Fail", "Fail", "Fail","Fail","Fail", "Fail","Fail","Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass"]
    
}

df = pd.DataFrame(data)

print(df)

x = df[["study_hours","attendance"]]
y = df["result"]

model = DecisionTreeClassifier(criterion="gini")

model.fit(x,y)

new_student = pd.DataFrame({
    
    "study_hours" : [5], 
    "attendance" : [85]
})

prediction = model.predict(new_student)
print(prediction)
print(model.score(x,y))
