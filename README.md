# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: S KAVIYA
RegisterNumber:  212223040090
*/

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
# Data
![image](https://github.com/user-attachments/assets/0b7979a6-3e0c-4960-af5d-66b03bdd7190)

# data.shape()
![image](https://github.com/user-attachments/assets/0bca540e-4abf-41c1-8e64-867f1ec05799)

# x.shape()
![image](https://github.com/user-attachments/assets/9192d1fa-79fa-4319-a3b1-2a5b9cf5d172)

# y.shape()
![image](https://github.com/user-attachments/assets/751eca52-db41-4972-ba08-12a5e1db4a0f)

# x_train.shape()
![image](https://github.com/user-attachments/assets/8eb2cbd4-6722-40be-bca5-8d9825b2c730)

# y_pred
![image](https://github.com/user-attachments/assets/2b0252dd-7fd9-42e7-876f-67dbc40a1162)

# acc(accuracy)
![image](https://github.com/user-attachments/assets/8cb2b923-8bf4-4c3d-b170-472b708e8414)

# con(confusion matrix)
![image](https://github.com/user-attachments/assets/647fc41d-c0c9-488a-97b6-f03405222c47)

# cl(classification report)
![image](https://github.com/user-attachments/assets/9b3a59f0-2aeb-4216-ad08-a5a1f2c2732c)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
