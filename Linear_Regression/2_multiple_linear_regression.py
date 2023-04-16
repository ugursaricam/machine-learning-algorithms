import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv('datasets/advertising.csv')

df.head()

#       TV  radio  newspaper  sales
# 0 230.10  37.80      69.20  22.10
# 1  44.50  39.30      45.10  10.40
# 2  17.20  45.90      69.30   9.30
# 3 151.50  41.30      58.50  18.50
# 4 180.80  10.80      58.40  12.90

df.shape # (200, 4)
df.info()

#  #   Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   TV         200 non-null    float64
#  1   radio      200 non-null    float64
#  2   newspaper  200 non-null    float64
#  3   sales      200 non-null    float64

df.isnull().sum().sum() # 0
df.describe().T

#            count   mean   std  min   25%    50%    75%    max
# TV        200.00 147.04 85.85 0.70 74.38 149.75 218.82 296.40
# radio     200.00  23.26 14.85 0.00  9.97  22.90  36.52  49.60
# newspaper 200.00  30.55 21.78 0.30 12.75  25.75  45.10 114.00
# sales     200.00  14.02  5.22 1.60 10.38  12.90  17.40  27.00

########################################
# Multiple Linear Regression
########################################

X = df.drop('sales', axis=1)
y = df[['sales']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=1)

X_train.shape # (160, 3)
X_test.shape # (40, 3)
y_train.shape # (160, 1)
y_test.shape # (40, 3)

lr_model = LinearRegression().fit(X_train, y_train)

# y = b + w1*x + w2*x + w3*x ...

# intercept (b - bias)
lr_model.intercept_[0] # 2.9079470208164295

# ceoficient (w - weight)
lr_model.coef_[0] # 0.0468431 , 0.17854434, 0.00258619

########################################
# Manual prediction
########################################

# TV = 30.00
# radio = 10.00
# newspaper = 40.00
# sales = ?

# y = b + w1*x + w2*x + w3*x ...

y_manual_pred = lr_model.intercept_[0] + lr_model.coef_[0][0]*30 + lr_model.coef_[0][1]*10 + lr_model.coef_[0][2]*40
y_manual_pred # 6.202130997974463

new_data = [[30], [10], [40]]
new_data = pd.DataFrame(new_data, index=X.columns).T
lr_model.predict(new_data) # 6.202131

########################################
# Plotting
########################################

def lr_model_plot(dataframe, var1, var2):
    sns.regplot(x=df[var1], y=df[var2], scatter_kws={'color': 'b', 's': 9}, ci=False, color='r')
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.show(block=True)

for i in X.columns:
    lr_model_plot(df, i, 'sales')

########################################
# MSE -  Mean Squared Error
########################################

# train MSE
y_pred = lr_model.predict(X_train)
train_mse = mean_squared_error(y_train, y_pred)
train_mse # 3.0168306076596774

# test MSE
y_pred = lr_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
test_mse # 1.9918855518287892

########################################
# RMSE -  Root Mean Squared Error
########################################

# train RMSE
np.sqrt(train_mse) # 1.736902590147092

# test RMSE
np.sqrt(test_mse) # 1.4113417558581582

########################################
# MAE -  Mean Absolute Error
########################################

# train MAE
y_pred = lr_model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_pred)
train_mae # 1.3288502460998386

# test MAE
y_pred = lr_model.predict(X_test)
train_mae = mean_absolute_error(y_test, y_pred)
train_mae # 1.0402154012924716

########################################
# Accuracy score
########################################

# train acc. score
lr_model.score(X_train, y_train) # 0.8959372632325174

# test acc. score
lr_model.score(X_test, y_test) # 0.8927605914615384

########################################
# Cross validation
########################################

import sklearn
sklearn.metrics.get_scorer_names()

cv_results = cross_validate(lr_model,
                            X, y,
                            cv=5,
                            scoring=['neg_mean_squared_error'])

-(cv_results['test_neg_mean_squared_error'].mean()) # 3.072946597100212 MSE

np.sqrt(-(cv_results['test_neg_mean_squared_error'].mean())) # 1.7529822010220788 RMSE
