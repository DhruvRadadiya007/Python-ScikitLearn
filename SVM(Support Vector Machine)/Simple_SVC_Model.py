from sklearn.svm import SVC
import pandas as pd

data = {
"Marks":[10,20,30,40, 50,60,70],
"Result":["Fail","Fail", "Fail","Pass","Pass","Pass", "Pass"]
}

df = pd.DataFrame(data)
print(df)

x = df[["Marks"]]
y =df["Result"]

model = SVC()

model.fit(x,y)

print(model.predict([["25"]]))