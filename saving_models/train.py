import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle

df=pd.read_csv('D:\ML project (House price prediction)\cleaned_train_data.csv')
print(df.head())

X= df.drop(columns=['SalePrice'],axis=1)
y= df['SalePrice']


#Train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Transforming X_train first
categorical_features=X_train[['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition']]

numerical_features=X_train[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold']]


#Encoding the categorical features
encoder=OneHotEncoder(handle_unknown='ignore')
encoded_categorical =encoder.fit_transform(categorical_features).toarray()

# Scaling the numerical features
scaler=StandardScaler()
scaled_numerical= scaler.fit_transform(numerical_features)

#forming X_train_transformed
X_train_transformed=pd.concat(
    [pd.DataFrame(encoded_categorical),pd.DataFrame(scaled_numerical)],
     axis=1
)

#Transforming X_test now
categorical_features1=X_test[['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition']]

numerical_features1=X_test[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold']]

#Tranforming the categorical features in X_test
encoded_categorical1=encoder.transform(categorical_features1).toarray()

#Transforming the numerical features in X_test
scaled_numerical1=scaler.transform(numerical_features1)

#Forming X_test_transformed
X_test_transformed=pd.concat(
    [pd.DataFrame(encoded_categorical1),pd.DataFrame(scaled_numerical1)],axis=1
)

#Training the model
model=GradientBoostingRegressor(n_estimators=150,learning_rate= 0.1, max_depth= 5, min_samples_leaf= 2, min_samples_split= 5, subsample= 0.9)
model.fit(X_train_transformed,y_train)

#Making Predictions
testing_accuracy= model.score(X_test_transformed,y_test)
training_accuracy= model.score(X_train_transformed,y_train)
print(f'The accuracy is {testing_accuracy}')
print(training_accuracy)

# Saving the models in pickle files
pickle.dump(encoder,open('encoder.pkl','wb'))
pickle.dump(scaler,open('scaler.pkl','wb'))
pickle.dump(model,open('model.pkl','wb'))
