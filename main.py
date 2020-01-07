import os
import pandas as pd

data = pd.read_csv('./dataset.csv')

data['Type'].value_counts(dropna=False)

data['Parking'] = data['Parking'].fillna(0)
data['Bathroom'] = data['Bathroom'].fillna(0)
data['Furnishing'] = data['Furnishing'].fillna('Unfurnished')
data['Type'] = data['Type'].fillna('Apartment')
data = data.drop(columns=['Per_Sqft'])

data["Furnishing"] = data["Furnishing"].astype('category')
data["Transaction"] = data["Transaction"].astype('category')
data["Type"] = data["Type"].astype('category')
data["Status"] = data["Status"].astype('category')

data["Furnishing"] = data["Furnishing"].cat.codes
data["Transaction"] = data["Transaction"].cat.codes
data["Type"] = data["Type"].cat.codes
data["Status"] = data["Status"].cat.codes

data = data.drop(columns=['Locality'])

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2)
Y_train = train['Price']
X_train = train.drop(columns=['Price'])
Y_test = test['Price']
X_test = test.drop(columns=['Price'])

from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
reg = linear_model.ElasticNet(random_state=0)
reg.fit(X_train, Y_train)
mean_absolute_error(reg.predict(X_test), Y_test)

print(reg.coef_)


