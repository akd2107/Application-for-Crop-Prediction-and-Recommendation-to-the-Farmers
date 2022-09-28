

import pandas as pd
import numpy as np

agri = pd.read_csv("D:\Data Science\PROJECTS-INNODATATICS\AGRICULTURE-PROJECT//Agri_Data.csv")



agri = agri.iloc[:, 2:14] # Excluding id column
# check for count of NA'sin each column
agri.isna().sum()

# Create an imputer object that fills 'Nan' values
# Mean and Median imputer are used for numeric data 


# for Mean, Meadian, Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer

# Mean Imputer 
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
agri["Production"] = pd.DataFrame(mean_imputer.fit_transform(agri[["Production"]]))
agri["Production"].isna().sum()
agri["Rainfall"] = pd.DataFrame(mean_imputer.fit_transform(agri[["Rainfall"]]))
agri["Rainfall"].isna().sum()



agri.isna().sum()




# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
agri_n = norm_func(agri.iloc[:, 1:])
agri_n.describe()

X = np.array(agri_n.iloc[:,:]) # Predictors 
Y = np.array(agri['Crop']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.6)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 21)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 


# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
