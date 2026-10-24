import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# creating a csv file


data = {
 
    'Hours Of Study': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4],
    'Marks': [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30, 24, 67, 69]

}

df = pd.DataFrame(data)

print(df)


df.to_csv('hours_marks.csv', index = False)

x = df[['Hours Of Study']]
y = df["Marks"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,)

model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(pd.DataFrame(x_test))
print(pd.DataFrame(y_pred))

plt.scatter(x,y,color = 'blue')
plt.plot(x,model.predict(x), color = 'red')
plt.xlabel('Hours of Study')
plt.ylabel('Marks')
plt.title('Linear Regression - Hours of Study vs Marks')
plt.show()


mse = mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error: {mse}")

