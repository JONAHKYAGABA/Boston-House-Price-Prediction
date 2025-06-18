# -*- coding: utf-8 -*-
"""
Boston House Price Prediction Project
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation

# Load dataset
BOSTON_DATA = datasets.load_boston()

# Convert dataset to DataFrame
def add_target_to_data(dataset):
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    print("Before adding target: ", df.shape)
    df['PRICE'] = dataset.target
    print("After adding target: {} \n {}\n".format(df.shape, df.head(2)))
    return df

# Plot regression graphs
def plotting_graph(df, features, n_row=2, n_col=5):
    fig, axes = plt.subplots(n_row, n_col, figsize=(16, 8))
    for i, feature in enumerate(features):
        row = int(i / n_col)
        col = i % n_col
        sns.regplot(x=feature, y='PRICE', data=df, ax=axes[row][col])
    plt.tight_layout()
    plt.show()

# Split dataset
def split_dataframe(df):
    label_data = df['PRICE']
    input_data = df.drop(['PRICE'], axis=1)
    input_train, input_eval, label_train, label_eval = train_test_split(input_data, label_data, test_size=0.3, random_state=42)
    return input_train, input_eval, label_train, label_eval

boston_df = add_target_to_data(BOSTON_DATA)
features = ['RM', 'ZN', 'INDUS', 'NOX', 'AGE', 'PTRATIO', 'LSTAT', 'RAD', 'CRIM', 'B']
plotting_graph(boston_df, features)

# Correlation heatmap
correlation_matrix = boston_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Linear Regression
X_train, X_test, Y_train, Y_test = split_dataframe(boston_df)
model = LinearRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
expectation = Y_test
lr_mse = mean_squared_error(expectation, pred)
lr_rmse = np.sqrt(lr_mse)

print('LR_MSE: {0:.3f}, LR_RMSE: {1:.3f}'.format(lr_mse, lr_rmse))
print('Regression Coefficients:', np.round(model.coef_, 1))
coeff = pd.Series(data=model.coef_, index=X_train.columns).sort_values(ascending=False)
print(coeff)

plt.scatter(expectation, pred)
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel('Expected price')
plt.ylabel('Predicted price')
plt.tight_layout()
plt.show()

# Additional Regressors
models = {
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "SGD": SGDRegressor(max_iter=1000, tol=1e-3),
    "XGB": XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                         max_depth=5, alpha=10, n_estimators=10)
}

pred_record = {}
for name, model in models.items():
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, pred)
    rmse = np.sqrt(mse)
    print(f"{name} - MSE: {mse:.3f}, RMSE: {rmse:.3f}")
    pred_record[name] = pred

# Plot example
for name in ["SGD", "XGB", "Gradient Boosting"]:
    prediction = pred_record[name]
    plt.scatter(expectation, prediction)
    plt.plot([0, 50], [0, 50], '--k')
    plt.xlabel('Expected price')
    plt.ylabel(f'Predicted price ({name})')
    plt.title(name)
    plt.tight_layout()
    plt.show()

# Neural Network
model = keras.Sequential([
    Dense(512, input_dim=BOSTON_DATA.data.shape[1]), Activation('relu'),
    Dense(256), Activation('relu'),
    Dense(128), Activation('relu'),
    Dense(64), Activation('relu'),
    Dense(1)
])

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, epochs=100, verbose=0)
loss, test_acc = model.evaluate(X_test, Y_test)
print('Neural Net - Test Loss : {:.4f} | Test Accuracy : {}'.format(loss, test_acc))

prediction = model.predict(X_test)
plt.scatter(expectation, prediction)
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel('Expected price')
plt.ylabel('Predicted price (Neural Net)')
plt.tight_layout()
plt.show()
