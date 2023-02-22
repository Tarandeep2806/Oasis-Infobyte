# -*- coding: utf-8 -*-

# -- Sheet --

# # IRIS FLOWER CLASSIFICATION
# #### ML Based Learning Model identifying the class of a model
# 
# Editor : Tarandeep Singh Gujral
# 
# Date : February 2023


# ### Importing Necessary Libraries


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import os

# ### Loading the Dataset


#loading the dataset
df= pd.read_csv('Iris.csv')
print("Top 10 rows of dataset:")
print(df.head(10))

# ### Description and Info about data   


print("Checking Null Values and Datatype of Dataset")
df.info()
print("\nStatistic of Dataset")
df.describe()
#print("\nNo. of samples in each species")
#df['Species'].value_counts()

# #### No. of samples in each species


df['Species'].value_counts()

# # EDA of Dataset


df['SepalLengthCm'].hist(alpha=0.6, color='blue')
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel('Count')
plt.show()

df['SepalWidthCm'].hist(alpha=0.6, color='green')
plt.title("Histogram of Sepal Width")
plt.xlabel("Sepal Width")
plt.ylabel('Count')
plt.show()

df['PetalLengthCm'].hist(alpha=0.6, color='orange')
plt.title("Histogram of Petal Length")
plt.xlabel("Petal Length")
plt.ylabel('Count')
plt.show()

df['PetalWidthCm'].hist(alpha=0.6, color='purple')
plt.title("Histogram of Petal Width")
plt.xlabel("Petal Width")
plt.ylabel('Count')
plt.show()

color = ['red','black','green']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'],c =color[i], label=species[i])
plt.xlabel("Sepal Lenth")
plt.ylabel("Sepal Width")
plt.legend()

color = ['red','black','green']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'],c =color[i], label=species[i])
plt.xlabel("Petal Lenth")
plt.ylabel("Petal Width")
plt.legend()

color = ['red','black','green']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'],x['PetalLengthCm'],c =color[i], label=species[i])
plt.ylabel("Petal Lenth")
plt.xlabel("Sepal Length")
plt.legend()

color = ['red','black','green']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'],x['PetalWidthCm'],c =color[i], label=species[i])
plt.ylabel("Petal Width")
plt.xlabel("Sepal Width")
plt.legend()

# ## Correlation of Matrix


df.corr()

corr = df.corr()
fig, ax = plt.subplots(figsize=(10,10))
sb.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Species'] = le.fit_transform(df['Species'])
df.head()

# ## Model Training


from sklearn.model_selection import train_test_split
X = df.drop(columns = ['Species'])
Y = df['Species'] 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(x_train, y_train)

print("Accuracy: ", model.score(x_test,y_test)*100)

from sklearn.neighbors import KNeighborsClassifier
model= KNeighborsClassifier()

model.fit(x_train,y_train)

print("Accuracy: ", model.score(x_test,y_test)*100)



