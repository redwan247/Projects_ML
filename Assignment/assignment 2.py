import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("breast_cancer_data.csv")

count = 0
for col in dataset.columns:
    print(col)
    count+=1
    
print("\n#Number of features: ", count)

dataset['diagnosis'].value_counts()

dataset.isnull().sum()

dataset = dataset.dropna(axis=1)

dataset.drop("id", axis=1, inplace=True)

X = dataset.drop("diagnosis", axis=1)
y = dataset["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

def models(X_train,Y_train):
    #Logistic Regression
    modelLRC = LogisticRegression()
    modelLRC.fit(X_train, Y_train)
    
    #KNN
    modelKNNC = KNeighborsClassifier(n_neighbors = 3)
    modelKNNC.fit(X_train, Y_train)
    
    #SVM
    modelSVMC = SVC(kernel = 'linear')
    modelSVMC.fit(X_train, Y_train)
    
    #Decision Tree
    modelDTC = DecisionTreeClassifier()
    modelDTC.fit(X_train, Y_train)
    
    #Naive Bayes
    modelNBC = GaussianNB()
    modelNBC.fit(X_train, Y_train)
    
    return modelLRC, modelKNNC, modelSVMC, modelDTC, modelNBC

    model = models(X_train,y_train)

    for i in range(len(model)):
    print('Model', i)
    prediction = model[i].predict(X_test)
    print(prediction)
    print()

    modelLRC = LogisticRegression()
modelKNNC = KNeighborsClassifier(n_neighbors = 3)
modelSVMC = SVC(kernel = 'linear')
modelDTC = DecisionTreeClassifier()
modelNBC = GaussianNB()
    
modelLRC.fit(X_train, y_train)
modelKNNC.fit(X_train, y_train)
modelSVMC.fit(X_train, y_train)
modelDTC.fit(X_train, y_train)
modelNBC.fit(X_train, y_train)

LR = modelLRC.score(X_train, y_train)*100
KNN = modelKNNC.score(X_train, y_train)*100
SVM = modelSVMC.score(X_train, y_train)*100
DT = modelDTC.score(X_train, y_train)*100
NB = modelNBC.score(X_train, y_train)*100

for i in range(len(model)):
    cnf_matrix = confusion_matrix(y_test, model[i].predict(X_test))
    print('Model :', i)
    print(cnf_matrix)
    print()

print('#Accuracy chart: ')
print()
print('[0]Logistic Regression Training Accuracy:', LR)
print('[1]K Nearest Neighbor Training Accuracy', KNN)
print('[2]Support Vector Machine Training Accuracy:', SVM)
print('[3]Decision Tree Training Accuracy:', DT)
print('[4]Naive Bayes Training Accuracy:', NB)

if((LR>SVM) and (LR>KNN) and (LR>DT) and (LR>NB)) :
    print("Logistic Regression has highest accuracy: ", LR)
    
elif((KNN>SVM) and (KNN>LR) and (KNN>DT) and (KNN>NB)):
    print("K-Nearest Neighbor has highest accuracy: ", KNN)
    
elif((SVM>LR) and (SVM>KNN) and (SVM>DT) and (SVM>NB)):
    print("Support Vector Machine has highest accuracy: ", SVM)
    
elif((DT>LR) and (DT>KNN) and (DT>SVM) and (DT>NB)):
    print("Decision Tree has highest accuracy: ", DT)

else:
    print("Naive Bayes has highest accuracy: ", NB)