from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn.decomposition import PCA

# Load the saved model, encoder, and scaler
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
pca= pickle.load(open('pca_model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__,template_folder=r'D:/ML project (House price prediction)/deployment/templates')

# Define the home route
@app.route('/')
def home():
    return render_template('home.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_data = {
    'MSZoning': [request.form.get('MSZoning')],
    'Street': [request.form.get('Street')],
    'Alley': [request.form.get('Alley')],
    'LotShape': [request.form.get('LotShape')],
    'LandContour': [request.form.get('LandContour')],
    'Utilities': [request.form.get('Utilities')],
    'LotConfig': [request.form.get('LotConfig')],
    'LandSlope': [request.form.get('LandSlope')],
    'Neighborhood': [request.form.get('Neighborhood')],
    'Condition1': [request.form.get('Condition1')],
    'Condition2': [request.form.get('Condition2')],
    'BldgType': [request.form.get('BldgType')],
    'HouseStyle': [request.form.get('HouseStyle')],
    'RoofStyle': [request.form.get('RoofStyle')],
    'RoofMatl': [request.form.get('RoofMatl')],
    'Exterior1st': [request.form.get('Exterior1st')],
    'Exterior2nd': [request.form.get('Exterior2nd')],
    'MasVnrType': [request.form.get('MasVnrType')],
    'ExterQual': [request.form.get('ExterQual')],
    'ExterCond': [request.form.get('ExterCond')],
    'Foundation': [request.form.get('Foundation')],
    'BsmtQual': [request.form.get('BsmtQual')],
    'BsmtCond': [request.form.get('BsmtCond')],
    'BsmtExposure': [request.form.get('BsmtExposure')],
    'BsmtFinType1': [request.form.get('BsmtFinType1')],
    'BsmtFinType2': [request.form.get('BsmtFinType2')],
    'Heating': [request.form.get('Heating')],
    'HeatingQC': [request.form.get('HeatingQC')],
    'CentralAir': [request.form.get('CentralAir')],
    'Electrical': [request.form.get('Electrical')],
    'KitchenQual': [request.form.get('KitchenQual')],
    'Functional': [request.form.get('Functional')],
    'FireplaceQu': [request.form.get('FireplaceQu')],
    'GarageType': [request.form.get('GarageType')],
    'GarageFinish': [request.form.get('GarageFinish')],
    'GarageQual': [request.form.get('GarageQual')],
    'GarageCond': [request.form.get('GarageCond')],
    'PavedDrive': [request.form.get('PavedDrive')],
    'PoolQC': [request.form.get('PoolQC')],
    'Fence': [request.form.get('Fence')],
    'MiscFeature': [request.form.get('MiscFeature')],
    'SaleType': [request.form.get('SaleType')],
    'SaleCondition': [request.form.get('SaleCondition')],
    'MSSubClass': [int(request.form.get('MSSubClass', 0))],
    'LotFrontage': [int(request.form.get('LotFrontage', 0))],
    'LotArea': [int(request.form.get('LotArea', 0))],
    'OverallQual': [int(request.form.get('OverallQual', 0))],
    'OverallCond': [int(request.form.get('OverallCond', 0))],
    'YearBuilt': [int(request.form.get('YearBuilt', 0))],
    'YearRemodAdd': [int(request.form.get('YearRemodAdd', 0))],
    'MasVnrArea': [int(request.form.get('MasVnrArea', 0))],
    'BsmtFinSF1': [int(request.form.get('BsmtFinSF1', 0))],
    'BsmtFinSF2': [int(request.form.get('BsmtFinSF2', 0))],
    'BsmtUnfSF': [int(request.form.get('BsmtUnfSF', 0))],
    'TotalBsmtSF': [int(request.form.get('TotalBsmtSF', 0))],
    '1stFlrSF': [int(request.form.get('1stFlrSF', 0))],
    '2ndFlrSF': [int(request.form.get('2ndFlrSF', 0))],
    'LowQualFinSF': [int(request.form.get('LowQualFinSF', 0))],
    'GrLivArea': [int(request.form.get('GrLivArea', 0))],
    'BsmtFullBath': [int(request.form.get('BsmtFullBath', 0))],
    'BsmtHalfBath': [int(request.form.get('BsmtHalfBath', 0))],
    'FullBath': [int(request.form.get('FullBath', 0))],
    'HalfBath': [int(request.form.get('HalfBath', 0))],
    'BedroomAbvGr': [int(request.form.get('BedroomAbvGr', 0))],
    'KitchenAbvGr': [int(request.form.get('KitchenAbvGr', 0))],
    'TotRmsAbvGrd': [int(request.form.get('TotRmsAbvGrd', 0))],
    'Fireplaces': [int(request.form.get('Fireplaces', 0))],
    'GarageYrBlt': [int(request.form.get('GarageYrBlt', 0))],
    'GarageCars': [int(request.form.get('GarageCars', 0))],
    'GarageArea': [int(request.form.get('GarageArea', 0))],
    'WoodDeckSF': [int(request.form.get('WoodDeckSF', 0))],
    'OpenPorchSF': [int(request.form.get('OpenPorchSF', 0))],
    'EnclosedPorch': [int(request.form.get('EnclosedPorch', 0))],
    '3SsnPorch': [int(request.form.get('3SsnPorch', 0))],
    'ScreenPorch': [int(request.form.get('ScreenPorch', 0))],
    'PoolArea': [int(request.form.get('PoolArea', 0))],
    'MiscVal': [int(request.form.get('MiscVal', 0))],
    'MoSold': [int(request.form.get('MoSold', 0))],
    'YrSold': [int(request.form.get('YrSold', 0))]
    }


    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)

    # Process input data
    categorical_features = input_df[['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition']]
    numerical_features = input_df[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold']]

    # Encode and scale
    encoded_categorical = encoder.transform(categorical_features).toarray()
    scaled_numerical = scaler.transform(numerical_features)

    # Combine features
    processed_features = pd.concat(
        [pd.DataFrame(encoded_categorical), pd.DataFrame(scaled_numerical)],
        axis=1
    )

    processed_features_pca=pca.transform(processed_features)

    # Make prediction
    prediction = model.predict(processed_features_pca)[0]

    # Return the result
    return render_template('home.html', prediction_text=f'The predicted sales price is: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)