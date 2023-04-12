import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

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
# Simple Linear Regression
########################################

X = df[["TV"]] # independent variable
y = df[["sales"]] # dependent variable

lr_model = LinearRegression().fit(X,y)

# y = b + w*x

# intercept (b - bias)
lr_model.intercept_[0] # 7.032593549127693

# ceoficient (w - weight)
lr_model.coef_[0][0] # 0.047536640433019764

########################################
# Manual prediction
########################################

# TV = 150.00
# sales = ?

# y = b + w*x

y_manual_pred = lr_model.intercept_[0] + (lr_model.coef_[0][0] * 150)
y_manual_pred # 14.163089614080658

########################################
# MSE -  Mean Squared Error
########################################

y_pred = lr_model.predict(X)

mse = mean_squared_error(y, y_pred)
mse # # 10.512652915656757

y.mean() # 14.02
y.std() # 5.22

########################################
# RMSE -  Root Mean Squared Error
########################################

np.sqrt(mse) # 3.2423221486546887

########################################
# MAE -  Mean Absolute Error
########################################

mae = mean_absolute_error(y, y_pred)
mae # 2.549806038927486

########################################
# Accuracy score
########################################

lr_model.score(X, y) # 0.611875050850071

########################################
# Visualization of the model
########################################

g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9}, ci=False, color="r")
g.set_title(f"Model Formula: Sales = {round(lr_model.intercept_[0], 2)} + TV*{round(lr_model.coef_[0][0], 2)}")
g.set_ylabel("Number of Sales")
g.set_xlabel("TV add")
plt.xlim(-10)
plt.ylim(0)
plt.show(block=True)