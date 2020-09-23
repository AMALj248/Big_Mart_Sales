import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statistics as sts
#LOADING THE DATASETS AND COMBINIG IT INTO ONE
test = pd.read_csv('TestNew.csv')
train = pd.read_csv('Train.csv')

data = pd.concat([train,test],ignore_index=True, sort = False)
#data =train
#CHECKING NULL VALUES
print(data.isnull())
print(data.head(10))

#CLEANING THE DATA

#dropping identifiers that are not needed
data.drop(['Item_Identifier', 'Outlet_Identifier' , 'Outlet_Establishment_Year'],axis = 1 , inplace = True )
print(data.head(10))

#dropping other values that are not required
#data.drop(['Item_Type','Outlet_Type','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Location_Type','Outlet_Size'],axis=1,inplace=True)

#CONVERTING CATEGORICAL DATA INTO NUMERICAL


#cleaning the NaN values in Outlet_Type
#this will replace all NaN values with the mean of the non null values
mean_value=data['Item_Weight'].mean()
data['Item_Weight']=data['Item_Weight'].fillna(mean_value)
print(data['Item_Weight'].isnull())

print(data.Item_Fat_Content.unique())
#CONVERTING FAT CONTENT
data['Item_Fat_Content'].replace('Low Fat',0, inplace=True)
data['Item_Fat_Content'].replace('low fat',0, inplace=True)
data['Item_Fat_Content'].replace('LF',0, inplace=True)
data['Item_Fat_Content'].replace('Regular',1, inplace=True)
data['Item_Fat_Content'].replace('reg',1, inplace=True)

print(data.head(5))

print(data.Item_Type.unique())

#CONVERTING Item_Type
change= {'Dairy' : 1 , 'Soft Drinks' : 2, 'Meat' :3 , 'Fruits and Vegetables' : 4,  'Household' : 5 ,
 'Baking Goods' : 6, 'Snack Foods' : 7 , 'Frozen Foods' : 8 , 'Breakfast' :9 ,
 'Health and Hygiene' :10 , 'Hard Drinks': 11 , 'Canned':12 , 'Breads' :13,  'Starchy Foods' :14 ,
 'Others' :20 , 'Seafood' :15 }

data['Item_Type'] = data['Item_Type'].map(change)

print(data.iloc[:,3])
print(data['Item_Type'].isnull())

#cleaning the NaN values in Outlet_Type
#this will replace all NaN values with the mean of the non null values
mean_value=data['Item_Type'].mean()
data['Item_Type']=data['Item_Type'].fillna(mean_value)
print(data['Item_Type'].isnull())


#CONVERTING Outlet_Size
print(data.Outlet_Size.unique())
data['Outlet_Size'].replace('Small',0, inplace=True)
data['Outlet_Size'].replace('Medium',1, inplace=True)
data['Outlet_Size'].replace('High',2, inplace=True)
print(data.iloc[:,5])

#cleaning the NaN values in Outlet_Size
#this will replace all NaN values with the mean of the non null values
mean_value=data['Outlet_Size'].mean()
data['Outlet_Size']=data['Outlet_Size'].fillna(mean_value)
print(data['Outlet_Size'].isnull())



#CONVERTING Outlet_Location_Type
print(data.Outlet_Location_Type.unique())
print(data['Outlet_Location_Type'].isnull()) # no null values

data['Outlet_Location_Type'].replace('Tier 1',0, inplace=True)
data['Outlet_Location_Type'].replace('Tier 2',1, inplace=True)
data['Outlet_Location_Type'].replace('Tier 3',2, inplace=True)
print(data.iloc[:,6])

#CONVERTING Outlet_Type
print(data.Outlet_Type.unique())
print(data['Outlet_Type'].isnull())

change= {'Supermarket Type1' : 1 ,  'Supermarket Type2': 2 , 'Grocery Store' : 4 , 'Supermarket Type3' : 3}

data['Outlet_Type'] = data['Outlet_Type'].map(change)

print(data.iloc[:,7])
print(data['Outlet_Type'].isnull())

#plotting the data for better understanding

#from sklearn.decomposition import PCA
#pca = PCA(n_components=4)  # I have selected 7 components to test as main features

#pca.fit(data)
#print(pca.components_)
#print(pca.explained_variance_)
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance');

#passing the input and output data to x nd y
x = data.drop(['Item_Outlet_Sales'],axis = 1)
y = data['Item_Outlet_Sales']
print(data.isnull().sum())
#22222222222222222222222222222222222222222222222222222
# SPLITTING THE DATA


#dropping identifiers that are not needed
test.drop(['Item_Identifier', 'Outlet_Identifier' , 'Outlet_Establishment_Year'],axis = 1 , inplace = True )
print(test.head(10))

#dropping other values that are not required
#test.drop(['Item_Type','Outlet_Type','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Location_Type','Outlet_Size'],axis=1,inplace=True)

#CONVERTING CATEGORICAL test INTO NUMERICAL


#cleaning the NaN values in Outlet_Type
#this will replace all NaN values with the mean of the non null values
mean_value=test['Item_Weight'].mean()
test['Item_Weight']=test['Item_Weight'].fillna(mean_value)
print(test['Item_Weight'].isnull())

print(test.Item_Fat_Content.unique())
#CONVERTING FAT CONTENT
test['Item_Fat_Content'].replace('Low Fat',0, inplace=True)
test['Item_Fat_Content'].replace('low fat',0, inplace=True)
test['Item_Fat_Content'].replace('LF',0, inplace=True)
test['Item_Fat_Content'].replace('Regular',1, inplace=True)
test['Item_Fat_Content'].replace('reg',1, inplace=True)

print(test.head(5))

print(test.Item_Type.unique())

#CONVERTING Item_Type
change= {'Dairy' : 1 , 'Soft Drinks' : 2, 'Meat' :3 , 'Fruits and Vegetables' : 4,  'Household' : 5 ,
 'Baking Goods' : 6, 'Snack Foods' : 7 , 'Frozen Foods' : 8 , 'Breakfast' :9 ,
 'Health and Hygiene' :10 , 'Hard Drinks': 11 , 'Canned':12 , 'Breads' :13,  'Starchy Foods' :14 ,
 'Others' :20 , 'Seafood' :15 }

test['Item_Type'] = test['Item_Type'].map(change)

print(test.iloc[:,3])
print(test['Item_Type'].isnull())

#cleaning the NaN values in Outlet_Type
#this will replace all NaN values with the mean of the non null values
mean_value=test['Item_Type'].mean()
test['Item_Type']=test['Item_Type'].fillna(mean_value)
print(test['Item_Type'].isnull())


#CONVERTING Outlet_Size
print(test.Outlet_Size.unique())
test['Outlet_Size'].replace('Small',0, inplace=True)
test['Outlet_Size'].replace('Medium',1, inplace=True)
test['Outlet_Size'].replace('High',2, inplace=True)
print(test.iloc[:,5])

#cleaning the NaN values in Outlet_Size
#this will replace all NaN values with the mean of the non null values
mean_value=test['Outlet_Size'].mean()
test['Outlet_Size']=test['Outlet_Size'].fillna(mean_value)
print(test['Outlet_Size'].isnull())



#CONVERTING Outlet_Location_Type
print(test.Outlet_Location_Type.unique())
print(test['Outlet_Location_Type'].isnull()) # no null values

test['Outlet_Location_Type'].replace('Tier 1',0, inplace=True)
test['Outlet_Location_Type'].replace('Tier 2',1, inplace=True)
test['Outlet_Location_Type'].replace('Tier 3',2, inplace=True)
print(test.iloc[:,6])

#CONVERTING Outlet_Type
print(test.Outlet_Type.unique())
print(test['Outlet_Type'].isnull())

change= {'Supermarket Type1' : 1 ,  'Supermarket Type2': 2 , 'Grocery Store' : 4 , 'Supermarket Type3' : 3}

test['Outlet_Type'] = test['Outlet_Type'].map(change)

print(test.iloc[:,7])
print(test['Outlet_Type'].isnull())

x_train = data.drop(['Item_Outlet_Sales'],axis = 1)
y_train = data['Item_Outlet_Sales']
x_test = test.drop(['Item_Outlet_Sales'],axis = 1)
y_test = test['Item_Outlet_Sales']

from sklearn.linear_model import LinearRegression
ln_rg = LinearRegression()
ln_rg.fit(x_train, y_train)

#Predict test set values with the test set
ln_rg_pred = ln_rg.predict(x_test)
y_pred= ln_rg_pred

#plotting the scatter plot  between y_actual and y_predicited
plt.scatter(y_test, y_pred , c='green')
plt.xlabel;("Input Weight")
plt.ylabel("predicted Weight ")
plt.title(" True vs Predicted value : Linear Regression ")
plt.show()

lgr_score_train=ln_rg.score(x_train , y_train)
print("Training Score using LinearRegression:",lgr_score_train )
lgr_score_test = ln_rg.score(x_test , y_test)
print("Testing Score LinearRegression : ",lgr_score_test)