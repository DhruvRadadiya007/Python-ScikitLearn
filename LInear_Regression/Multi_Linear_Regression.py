from sklearn.linear_model import LinearRegression
import pandas as pd


df = pd.read_csv("LInear_Regression\Student_Performance.csv")
print(df.head(10))

X = df[["Hours Studied","Previous Scores","Sleep Hours","Sample Question Papers Practiced"]]
y = df["Performance Index"]


model = LinearRegression()

model.fit(X,y)

new_student = [[7,80,7,5]]
print(model.predict(new_student))
