import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

def train_logistic_regression(X_train, y_train):
    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}
    grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'models/logistic_regression.pkl')
    return best_model

def train_decision_tree(X_train, y_train):
    param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30, 40, 50], 
                  'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'models/decision_tree.pkl')
    return best_model

def train_knn(X_train, y_train):
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski']}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'models/knn.pkl')
    return best_model

def train_naive_bayes(X_train, y_train):
    param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
    grid_search = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'models/naive_bayes.pkl')
    return best_model

if __name__ == "__main__":
    from data_preparation import load_and_prepare_data
    X_train_scaled, X_test_scaled, y_train, y_test, data = load_and_prepare_data()
    
    train_logistic_regression(X_train_scaled, y_train)
    train_decision_tree(X_train_scaled, y_train)
    train_knn(X_train_scaled, y_train)
    train_naive_bayes(X_train_scaled, y_train)
