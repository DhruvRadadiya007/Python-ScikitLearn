from sklearn.linear_model import LinearRegression

import pandas as pd


# data = {
    
#     'Hours' : [1,2,3,4,5,6,7,8],
#     'Marks' : [35,40,45,50,55,60,65,70]
     
# } # sample data set

# df = pd.DataFrame(data) # converting the dataset to data frame formate using pandas
# print(df) 

# # defining x and y variables
# X = df[["Hours"]] # we have to use two brackets here because it takes input as 2d matrix and out put as 1d
# y = df["Marks"]

# # creating a Linear regression model 

# model = LinearRegression() 

# model.fit(X,y) # providing the model variables 

# print(model.predict([[9]])) # perdict the marks if the number of input hours are ____

# # print(model.predict(pd.DataFrame({"Hours":[9]}))) # to ignore the warning using a dataframe formate






data = { 
      
"Area" : [500,700,800,1000,1200,1500,1800,2000],

"Price" : [20,28,32,40,48,60,72,80]

}

df = pd.DataFrame(data)
print(df)

x = df[["Area"]]
y = df["Price"]


model = LinearRegression()

model.fit(x,y)

print(model.predict([[2500]]))






















