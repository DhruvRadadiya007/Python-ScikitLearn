import pandas as pd
from sklearn.naive_bayes import GaussianNB

2

data = {
"StudyHours": [1,2,2,3,4,5,5,6,7,8],
"Attendance" : [35,45,50,55,60,70,75,80,90,95],
"Result":["Fail","Fail","Fail","Fail","Fail","Pass", "Pass", "Pass", "Pass", "Pass"]
}

df = pd.DataFrame(data)

x=df[["StudyHours","Attendance"]]
y = df["Result"]

model = GaussianNB()

model.fit(x,y)

new_student = pd. DataFrame({"StudyHours":[5.5],"Attendance":[78]})

print(model.predict(new_student))