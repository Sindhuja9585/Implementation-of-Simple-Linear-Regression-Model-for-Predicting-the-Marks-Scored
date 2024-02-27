# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.



## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SINDHUJA P
RegisterNumber:  212222220047
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

Dataset:

![282738473-05658adb-62cb-4564-96cf-bf93ee1cb0a1](https://github.com/Sindhuja9585/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860624/05ef62a4-0124-4eca-baa9-6a660a851005)

Head values:

![282738621-fc389afd-30f6-499c-8145-4df113de593e](https://github.com/Sindhuja9585/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860624/fa72015f-1889-40f4-b15c-6824d083903b)

Tail values:

![282738736-e43958b1-f0d9-4004-a769-a5a87d51e44f](https://github.com/Sindhuja9585/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860624/dc6b405a-7efc-42eb-ac31-b965a6b0a55f)

X and Y values:

![282738825-b1e35e5a-4533-437c-9596-fe57211ebd21](https://github.com/Sindhuja9585/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860624/aa82c94b-cc1a-4a27-81e7-6298b68636c5)

Predication values of X and Y:

![282738961-c85cfd52-7fc2-4450-a052-fa23642cd3f6](https://github.com/Sindhuja9585/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860624/0a3ccb6c-1a7a-4e49-b605-f3d799b0af64)

MSE,MAE and RMSE:

![282739072-e26c691a-f687-4931-815f-31e1a569b0c5](https://github.com/Sindhuja9585/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860624/b56add87-d82f-4efc-9bda-7e079a7ab791)

Training Set:

![282739253-ee2dde61-1636-43f1-9924-cd06af6aee99](https://github.com/Sindhuja9585/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860624/156fa1d5-b8ea-43fc-9466-d1a007f2c268)


Testing Set:

![282739482-7f86c66a-08f2-49f9-a87e-4e269b2f9f2f](https://github.com/Sindhuja9585/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122860624/21617e7e-cc4b-4380-9cc3-d96726676c95)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
