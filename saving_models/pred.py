import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the saved model, encoder, and scaler
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
pca= pickle.load(open('pca_model.pkl', 'rb'))

# Function to process new input data
def process_input_data(input_data):
    # Split categorical and numerical features
    categorical_features = input_data[['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition']]
    numerical_features = input_data[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold']]

    # One-Hot Encode categorical features
    encoded_categorical = encoder.transform(categorical_features).toarray()

    # Scale numerical features
    scaled_numerical = scaler.transform(numerical_features)

    # Combine processed features
    processed_features = pd.concat(
        [pd.DataFrame(encoded_categorical), pd.DataFrame(scaled_numerical)],
        axis=1
    )

    # Apply PCA transformation
    processed_features_pca = pca.transform(processed_features)

    return pd.DataFrame(processed_features_pca)

# Function to make predictions
def predict(input_data):
    # Process the input data
    processed_features = process_input_data(input_data)

    # Make predictions
    predictions = model.predict(processed_features)

    return predictions

# Example usage
if __name__ == "__main__":
    # Example input data
    example_data = pd.DataFrame({
    'MSZoning': ['RL'],
    'Street': ['Pave'],
    'Alley': ['Grvl'],
    'LotShape': ['Reg'],
    'LandContour': ['Lvl'],
    'Utilities': ['AllPub'],
    'LotConfig': ['Inside'],
    'LandSlope': ['Gtl'],
    'Neighborhood': ['CollgCr'],
    'Condition1': ['Norm'],
    'Condition2': ['Norm'],
    'BldgType': ['1Fam'],
    'HouseStyle': ['2Story'],
    'RoofStyle': ['Gable'],
    'RoofMatl': ['CompShg'],
    'Exterior1st': ['VinylSd'],
    'Exterior2nd': ['VinylSd'],
    'MasVnrType': ['BrkFace'],
    'ExterQual': ['Gd'],
    'ExterCond': ['TA'],
    'Foundation': ['PConc'],
    'BsmtQual': ['Gd'],
    'BsmtCond': ['TA'],
    'BsmtExposure': ['No'],
    'BsmtFinType1': ['GLQ'],
    'BsmtFinType2': ['Unf'],
    'Heating': ['GasA'],
    'HeatingQC': ['Ex'],
    'CentralAir': ['Y'],
    'Electrical': ['SBrkr'],
    'KitchenQual': ['Gd'],
    'Functional': ['Typ'],
    'FireplaceQu': ['Gd'],
    'GarageType': ['Attchd'],
    'GarageFinish': ['RFn'],
    'GarageQual': ['TA'],
    'GarageCond': ['TA'],
    'PavedDrive': ['Y'],
    'PoolQC': ['Gd'],
    'Fence': ['MnPrv'],
    'MiscFeature': ['Shed'],
    'SaleType': ['WD'],
    'SaleCondition': ['Normal'],
    # Numerical columns
    'MSSubClass': [20],
    'LotFrontage': [80.0],
    'LotArea': [9600],
    'OverallQual': [7],
    'OverallCond': [5],
    'YearBuilt': [2000],
    'YearRemodAdd': [2005],
    'MasVnrArea': [150.0],
    'BsmtFinSF1': [400],
    'BsmtFinSF2': [0],
    'BsmtUnfSF': [600],
    'TotalBsmtSF': [1000],
    '1stFlrSF': [1000],
    '2ndFlrSF': [800],
    'LowQualFinSF': [0],
    'GrLivArea': [1800],
    'BsmtFullBath': [1],
    'BsmtHalfBath': [0],
    'FullBath': [2],
    'HalfBath': [1],
    'BedroomAbvGr': [3],
    'KitchenAbvGr': [1],
    'TotRmsAbvGrd': [8],
    'Fireplaces': [1],
    'GarageYrBlt': [2000],
    'GarageCars': [2],
    'GarageArea': [500],
    'WoodDeckSF': [100],
    'OpenPorchSF': [50],
    'EnclosedPorch': [0],
    '3SsnPorch': [0],
    'ScreenPorch': [0],
    'PoolArea': [0],
    'MiscVal': [0],
    'MoSold': [6],
    'YrSold': [2007]
    })


    # Get predictions
    predictions = predict(example_data)

    print("Predictions:", predictions)