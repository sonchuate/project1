import pandas as pd
import numpy as np
#process
df = pd.read_csv('train.csv')
df['Cabin'].fillna(0, inplace = True)
df['Age'].fillna(df['Age'].median(), inplace= True)
df['Pclass'].fillna(df['Pclass'].median(), inplace= True)
df['SibSp'].fillna(df['SibSp'].median(), inplace= True)
df.fillna(0, inplace=True)
#choose feature and process data
data = np.array(df)
data = np.hstack((data, np.ones((data.shape[0],1))))
print(data)
print(df)
print("_______________________")
for i in data:
    if i[4] == 'female':
        i[4] = 1
    else:
        i[4] = 0
    #cabin
    if i[10] != 0:
        i[10] = 1    
    #Embarked
    if i[11] == 'C':
        i[11] = 1
    elif i[11] == 'S':
        i[11] = -1
    else:
        i[11] = 0

X = np.concatenate((data[:,2:5:2],data[:,5:8]),axis = 1)
X = np.concatenate((X,data[:,9:]),axis = 1)
y = data[:,1]
X = X.astype(float)
y = y.astype(float)

#create model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

clf = LogisticRegression(random_state=0, max_iter= 70).fit(X, y)
print(X.shape,y.shape)
y_predict = clf.predict(X)
print(y_predict)

df = pd.read_csv('test.csv')
df['Cabin'].fillna(0, inplace = True)
df['Age'].fillna(df['Age'].median(), inplace= True)
df['Pclass'].fillna(df['Pclass'].median(), inplace= True)
df['SibSp'].fillna(df['SibSp'].median(), inplace= True)
df.fillna(0, inplace=True)
print(df)
data_new = np.array(df)
data_new = np.hstack((data_new, np.ones((data_new.shape[0],1))))
for i in data_new:
    if i[3] == 'female':
        i[3] = 1
    else:
        i[3] = 0
    #cabin
    if i[9] != 0:
        i[9] = 1    
    #Embarked
    if i[10] == 'C':
        i[10] = 1
    elif i[10] == 'S':
        i[10] = -1
    else:
        i[10] = 0
X_new = np.concatenate((data_new[:,1:4:2],data_new[:,4:7]),axis = 1)
X_new = np.concatenate((X_new,data_new[:,8:]),axis = 1)
y_predict_new = clf.predict(X_new)
y_predict_new = y_predict_new.astype(int)
d = {
    "PassengerId" : pd.Series(df['PassengerId']) ,
    "Survived": pd.Series(y_predict_new)
}
df_new = pd.DataFrame(d)
df_new.to_csv('output.csv',index = None)