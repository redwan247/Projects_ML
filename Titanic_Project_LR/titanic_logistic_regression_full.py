import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline
import math
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

###import_data
titanic_data = pd.read_csv("Train_2.csv")
titanic_data.head(10)

print("# of passengers in the original data: "+str(len(titanic_data.index)))

###analyzing_data
sns.countplot(x="Survived", data=titanic_data)

sns.countplot(x="Survived", hue="Sex", data=titanic_data)

sns.countplot(x="Survived", hue="Pclass", data=titanic_data)

titanic_data["Age"].plot.hist()
titanic_data["Fare"].plot.hist(bins=20,figsize=(10,5))

titanic_data.info()

sns.countplot(x="SibSp", data=titanic_data)


###data_wrangling
titanic_data.isnull()
titanic_data.isnull().sum()

sns.heatmap(titanic_data.isnull(), yticklabels=False)
titanic_data.head(10)

titanic_data.drop("Cabin", axis=1, inplace=True)
titanic_data.head(10)

titanic_data.dropna(inplace=True)
sns.heatmap(titanic_data.isnull(), yticklabels=False)

titanic_data.isnull().sum()

sex = pd.get_dummies(titanic_data["Sex"], drop_first=True)
sex.head(5)

embark = pd.get_dummies(titanic_data["Embarked"], drop_first=True)
embark.head(5)

Pcl = pd.get_dummies(titanic_data["Pclass"], drop_first=True)
Pcl.head(5)

titanic_data = pd.concat([titanic_data, sex, embark, Pcl], axis=1)
titanic_data.head(5)

titanic_data.drop(['Sex', 'Embarked', 'PassengerId', 'Ticket', 'Name'], axis=1, inplace=True)
titanic_data.head()

titanic_data.drop('Pclass', axis=1, inplace=True)
titanic_data.head()


##train
X = titanic_data.drop("Survived", axis=1)
y = titanic_data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

prediction = logmodel.predict(X_test)
classification_report(y_test, prediction)

##accuracy
confusion_matrix(y_test,prediction)
accuracy_score(y_test,prediction)