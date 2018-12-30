import numpy as np
import pandas as pd
import os
print(os.listdir(r"C:\Users\onlyp\Documents\SUBLIME_TEXT_SAVES"))
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv(r'C:\Users\onlyp\Documents\SUBLIME_TEXT_SAVES\advertising.csv')
df.head()
df.describe()
df.info()
sns.set_style('whitegrid')
df['Age'].hist(bins=30)
sns.jointplot(x='Age', y='Area Income',data=df)
sns.jointplot(x='Age', y='Daily Time Spent on Site',data=df,kind='kde',color='red')
sns.jointplot(x='Daily Time Spent on Site' , y='Daily Internet Usage',data=df,color='green')
sns.pairplot(df,hue='Clicked on Ad',palette='bwr')
df['Timestamp']=pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].apply(lambda time: time.hour)
df['Month'] = df['Timestamp'].apply(lambda time: time.month)
df['Day of Week'] = df['Timestamp'].apply(lambda time: time.dayofweek)
df.tail(5)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop(['Clicked on Ad','Ad Topic Line','City','Country','Timestamp'],axis=1), 
                                                    df['Clicked on Ad'], test_size=0.33, 
                                                    random_state=42)
X_train.info()
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))