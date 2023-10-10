#IMPORTING THE REQUIRED LIBRARIES
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#IMPORTING THE DATASET
red_wine = pd.read_csv("D:\ML Project\wine+quality\winequality-red updated.csv")
white_wine = pd.read_csv("D:\ML Project\wine+quality\winequality-white updated.csv")

#SPLITTING THE DATA SET INTO TARGET VARIABLE AND INDEPENDENT VARIABLE FOR BOTH RED AND WHITE WINE
X = red_wine.iloc[:,:-1].values
y = red_wine.iloc[:,-1].values
X_1 = white_wine.iloc[:,:-1].values
y_1 = white_wine.iloc[:,-1].values 


#IMPORTING THE LIBRARIES FOR DATA VISUALIZATION
import matplotlib.pyplot as plt
import seaborn as sns

#VISUALIZING THE DATA USING COLOUR MAPS
colormap = plt.cm.RdYlBu                        #FOR RED WINE
fig, ax = plt.subplots(figsize=(10, 8))
plt.title('Feature Correlation of Red Wine', fontsize=15, pad=20)
corr_matrix = red_wine.corr()
sns.heatmap(corr_matrix, cmap=colormap, linewidths=0.1, annot=True, fmt=".2f", square=True, ax=ax)

colormap = plt.cm.coolwarm                      #FOR WHITE WINE
fig, ax = plt.subplots(figsize=(10, 8))
plt.title('Feature Correlation of White Wine', fontsize=15, pad=20)
corr_matrix = white_wine.corr()
sns.heatmap(corr_matrix, cmap=colormap, linewidths=0.1, annot=True, fmt=".2f", square=True, ax=ax)
plt.show()

#SPILTTING THE DATA INTO TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_1train, X_1test, y_1train, y_1test = train_test_split(X_1, y_1, test_size=0.2, random_state=42)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_1train = scaler.fit_transform(X_1train)
X_1test = scaler.transform(X_1test)


#RANDOM FOREST REGRESSOR TO TRAIN THE MODEL FOR RED WINE AND WHITE WINE
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor_1 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
rf_regressor_1.fit(X_1train, y_1train)
y_pred = rf_regressor.predict(X_test)
y_1pred = rf_regressor_1.predict(X_1test)

#CHECKING THE PRECISION OF OUR ML MODEL
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse_1 = mean_squared_error(y_1test, y_1pred)
mae_1 = mean_absolute_error(y_1test, y_1pred)
r2_1 = r2_score(y_1test, y_1pred)


print('For red wine : ')
print(f'Mean Squared Error: {mse:.2f}')
print(f'Mean Absolute Error: {mae:.2f}')
print(f'R-squared Score: {r2:.2f}')
print('For white wine:')
print(f'Mean Squared Error: {mse_1:.2f}')
print(f'Mean Absolute Error: {mae_1:.2f}')
print(f'R-squared Score: {r2_1:.2f}')

#COMPARING THE PREDICTED AND ACTUAL VALUES
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_pred),1)),1))


#CHECKING THE IMPORTANCE OF FEATURES AND DISPLAYING IT IN A SORTED MANNER FROM THE MOST RELEVANT 
#TO THE ONE WITH LEAST RELEVANCE TO THE QUALITY FACTOR OF WINE

feature_importance = rf_regressor.feature_importances_
feature_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                 "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                 "pH", "sulphates", "alcohol"]
idx = np.argsort(feature_importance)[::-1]

print("\nFeature Importance:")
for i in idx:
    print(f"{feature_names[i]}: {feature_importance[i]:.4f}")


feature_importance_1 = rf_regressor_1.feature_importances_
feature_names_1 = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                 "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                 "pH", "sulphates", "alcohol"]
idx_1 = np.argsort(feature_importance_1)[::-1]

print("\nFeature Importance:")
for i in idx_1:
    print(f"{feature_names_1[i]}: {feature_importance_1[i]:.4f}")


#CHECKING FOR OUTLIERS IN THE DATA i.e. POOR AND EXCELLENT WINES
poor = 4
excellent = 7 

poor_quality_outliers = red_wine[red_wine.iloc[:,-1].values < poor]
excellent_quality_outliers = red_wine[red_wine.iloc[:,-1].values > excellent]

poor_quality_outliers_ww = white_wine[white_wine.iloc[:,-1].values < poor]
excellent_quality_outliers_ww = white_wine[white_wine.iloc[:,-1].values > 8]


# print(poor_quality_outliers_ww)
# print(excellent_quality_outliers_ww)        
# print(poor_quality_outliers)
# print(excellent_quality_outliers)

# COMMENT OUT TO SHOW THE OUTLIERS