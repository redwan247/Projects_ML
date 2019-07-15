#importing

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


#loading_dataset

dataset = pd.read_csv("breast_cancer.csv")

dataset.head(10)

print("# of patient in the original data: "+str(len(dataset.index)))

dataset.dropna(inplace=True)

dataset.drop("Sample_code_number", axis=1, inplace=True)

dataset.head(5)

#train_test_split

X = dataset.drop("Class", axis=1)
y = dataset["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

#classifier_model

modelLRC = LogisticRegression()
modelSVMC  = svm.SVC(kernel='linear')
modelKNNC = KNeighborsClassifier(n_neighbors=3)

#fitting

modelLRC.fit(X_train, y_train)

modelSVMC.fit(X_train, y_train)

modelKNNC.fit(X_train, y_train)

#confusion_matrix

print("Confusion Matrix for Logistic Regression: ")
confusion_matrix(y_test,predictionLRC)

print("Confusion Matrix for Support Vector Machine: ")
confusion_matrix(y_test,predictionSVMC)

print("Confusion Matrix for KNN where (N=3): ")
confusion_matrix(y_test,predictionKNNC)

#accuracy

LR = accuracy_score(y_test,predictionLRC)*100
SVM = accuracy_score(y_test,predictionSVMC)*100
KNN = accuracy_score(y_test,predictionKNNC)*100

print("Accuracy Chart: ")
print("Logistic Regression: ",LR)
print("Support Vector Machine: ",SVM)
print("K-Nearest Neighbors (where k=3): ",KNN)

#soring

if((LR>SVM) and (LR>KNN)) :
    print("Logistic Regression has highest accuracy: ", LR)
    if(SVM>KNN):
        print("Support Vectro Machine stands in the middle: ", SVM)
        print("K-Nearest Neighbor has least accuracy: ", KNN)
    else:
        print("K-Nearest Neighbor stands in the middle: ", KNN)
        print("Support Vectro Machine has least accuracy: ", SVM)
elif((SVM>LR) and (SVM>KNN)):
    print("Support Vectro Machine has highest accuracy: ", SVM)
    if(LR>KNN):
        print("Logistic Regression stands in the middle: ", LR)
        print("K-Nearest Neighbor has least accuracy: ", KNN)
    else:
        print("K-Nearest Neighbor stands in the middle: ", KNN)
        print("Logistic Regression has least accuracy: ", LR)
else:
    print("K-Nearest Neighbor has highest accuracy: ", KNN)
    if(LR>SVM):
        print("Logistic Regression stands in the middle: ", LR)
        print("Support Vectro Machine has least accuracy: ", SVM)
    else:
        print("Support Vectro Machine stands in the middle: ", SVM)
        print("Logistic Regression has least accuracy: ", LR)