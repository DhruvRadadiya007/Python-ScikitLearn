import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data = pd.DataFrame(
    {
     'age':[20,24,28,32,34,38,42],
    'purchased_house':[0,0,1,0,1,1,1]     
    }
) # data set for classification

print(data)

# assigning the input variables
x = data[["age"]] 
y = data["purchased_house"]

# initialising the model
model = LogisticRegression()
model.fit(x,y)

# predicting the output
print(model.predict([[22]])) # 0 means false and 1 means True

# check the accuracy for the model

print(model.score(x,y))

# creating more ages by numpy

new_ages = np.linspace(18,45,100).reshape(-1,1)

# predicting the probabilities of all the ages between 0 to 1
y_prob = model.predict_proba(new_ages)[:,1]

print(y_prob)


# plotting the graph
plt.scatter(x,y,color="blue",label = "actual data")

plt.plot(new_ages,y_prob,color = "red",label = "Logistic regression curve")

plt.xlabel("Age")
plt.ylabel("Purchased House(probability)")
plt.title("Age vs House probabilities")

plt.show()

