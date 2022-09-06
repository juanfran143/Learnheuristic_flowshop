import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#from keras import models
#from keras import layers

def train_random_forest(fit = False):
    data = pd.read_table("data_base.txt",sep=" ")
    data_025 = data
    y = data_025["time_black_box"]
    x = data_025.loc[:, ["Machine_processed_jobs", "time_of_job_in_machine", "job_class"]]

    if fit:
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'bootstrap': [True],
            'max_depth': [10, 20, 30],
            'max_features': [2, 3],
            'min_samples_leaf': [4, 5],
            'n_estimators': [5, 10]
        }
        # Fit the random search model
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=5, verbose=2) #
        #{'bootstrap': True, 'max_depth': 30, 'max_features': 5, 'min_samples_leaf': 3,
        # 'min_samples_split': 4, 'n_estimators': 200}
        #

        #{'bootstrap': True, 'max_depth': 15, 'max_features': 3, 'min_samples_leaf': 1,
        # 'min_samples_split': 2, 'n_estimators': 250}
        grid_search.fit(x, y)
    #'bootstrap': True, 'max_depth': 10, 'max_features': 3, 'min_samples_leaf': 4, 'min_samples_split': 4, 'n_estimators': 150
    #'max_depth': 10, 'max_features': 3, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 30
    #'max_depth': 10, 'max_features': 3, 'min_samples_leaf': 4, 'min_samples_split': 4, 'n_estimators': 100
    #'max_depth': 20, 'max_features': 3, 'min_samples_leaf': 5, 'min_samples_split': 3, 'n_estimators': 250
    #'max_depth': 20, 'max_features': 3, 'min_samples_leaf': 4, 'min_samples_split': 3, 'n_estimators': 150
    #{'bootstrap': True, 'max_depth': 20, 'max_features': 3, 'min_samples_leaf': 4, 'min_samples_split': 3, 'n_estimators': 150}

    rf = RandomForestRegressor(bootstrap = True,  max_depth=10, max_features=2, min_samples_leaf=4,
                      min_samples_split=3, n_estimators=10, random_state=42)
    rf.fit(x.values, y.values)

    train_x = int(len(data_025) * 0.8)
    y_test = data_025.loc[train_x:,"time_black_box"]
    x_test = data_025.loc[train_x:, ["Machine_processed_jobs", "time_of_job_in_machine", "job_class"]]

    y_pred = rf.predict(x_test.values)
    print(mean_squared_error(y_test, y_pred))
    #rf.predict([x.loc[70]])

    #filename = 'finalized_model_025.sav'
    #pickle.dump(rf, open(filename, 'wb'))
    return rf

def train_decision_tree(fit = False):
    data = pd.read_table("data_base.txt",sep=" ")
    #data_025 = data[data.h == 2.5]
    data_025 = data
    y = data_025["time_black_box"]
    x = data_025.loc[:, ["Machine_processed_jobs", "time_of_job_in_machine", "job_class"]]
    x.columns = ["Machine_processed_jobs", "p'_ij", "job_class"]
    #x.columns = [""]
    if fit:
        rf = tree.DecisionTreeRegressor(random_state=42)
        param_grid = {
            'max_depth': [5, 10, 20, 30],
            'min_samples_leaf': [4, 5],
            'min_samples_split': [3, 4]
        }
        # Fit the random search model
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=5, verbose=2) #
        grid_search.fit(x, y)
    #RandomForestRegressor(max_depth=10, max_features=2, min_samples_leaf=4,
    #                  min_samples_split=3, n_estimators=50, random_state=42)
    rf = tree.DecisionTreeRegressor(max_depth = 10, min_samples_leaf = 5, min_samples_split= 3)

    train_x = int(len(data_025) * 0.8)
    y_test = data_025.loc[train_x:,"time_black_box"]
    x_test = data_025.loc[train_x:, ["Machine_processed_jobs", "time_of_job_in_machine", "job_class"]]

    rf.fit(x.values, y.values)
    y_pred = rf.predict(x_test.values)
    mean_squared_error(y_test, y_pred)
    #rf.predict([x.loc[70]])

    #filename = 'finalized_model_025.sav'
    #pickle.dump(rf, open(filename, 'wb'))
    tree.plot_tree(rf, feature_names=x.columns, class_names="Tiempo de procesamiento estimado", filled=True,
                   rounded=True)

    return rf

def train_linear_regression(fit = False):
    data_025 = pd.read_table("data_base.txt",sep=" ")

    train_x = int(len(data_025) * 0.8)
    y = data_025.loc[:train_x, "time_black_box"]
    x = data_025.loc[:train_x, ["Machine_processed_jobs", "time_of_job_in_machine", "job_class"]]

    y_test = data_025.loc[train_x:,"time_black_box"]
    x_test = data_025.loc[train_x:, ["Machine_processed_jobs", "time_of_job_in_machine", "job_class"]]

    regr = linear_model.LinearRegression()

    regr.fit(x, y)
    y_pred = regr.predict(x_test)

    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    adjusted_r_2 = 1 - (1-regr.score(x_test, y_test))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
    print("Coefficient of determination: %.2f" % adjusted_r_2)

    mean_squared_error(y_test, y_pred)

    return y_pred

if __name__ == "__main__":
    pass
