from helper_functions import *
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('datasets/diabetes.csv')

check_df(df)

################################################
# Data Preprocessing
################################################

y = df['Outcome']
X = df.drop(['Outcome'], axis=1)

df[X.columns] = StandardScaler().fit_transform(X)
X = df.drop(['Outcome'], axis=1)

################################################
# Modeling & Prediction
################################################

knn_model = KNeighborsClassifier().fit(X, y)
y_pred = knn_model.predict(X)

# for a random user
random_user = X.sample(1, random_state=45)
knn_model.predict(random_user) # array([1], dtype=int64)

################################################
# Model Evaluation
################################################

print(classification_report(y, y_pred))

#               precision    recall  f1-score   support
#            0       0.85      0.90      0.87       500
#            1       0.79      0.70      0.74       268
#     accuracy                           0.83       768
#    macro avg       0.82      0.80      0.81       768
# weighted avg       0.83      0.83      0.83       768


y_prob = knn_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob) # 0.9017686567164179

################################################
# Cross-Validation
################################################

cv_results = cross_validate(knn_model,
                            X, y,
                            cv=5,
                            scoring=['accuracy', 'f1', 'roc_auc'])

cv_results['test_accuracy'].mean() # 0.733112638994992
cv_results['test_f1'].mean() # 0.5905780011534191
cv_results['test_roc_auc'].mean() # 0.7805279524807827

################################################
# Hyperparameter Optimization
################################################

knn_model.get_params()

knn_params = {'n_neighbors' : range(2, 50)}

knn_params_opt = GridSearchCV(knn_model, knn_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)

knn_params_opt.best_params_ # {'n_neighbors': 17}

################################################
# Final Model
################################################

knn_final = knn_model.set_params(**knn_params_opt.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X, y,
                            cv=5,
                            scoring=['accuracy', 'f1', 'roc_auc'])

cv_results['test_accuracy'].mean() # 0.7669892199303965
cv_results['test_f1'].mean() # 0.6170909049720137
cv_results['test_roc_auc'].mean() # 0.8127938504542278
