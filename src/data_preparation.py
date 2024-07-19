import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_prepare_data():
    # Load the Breast Cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialise the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform the training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Use the same scaler to transform the test data
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, data

def visualize_data(X, y, data):
    df = pd.DataFrame(X, columns=data.feature_names)
    df['target'] = y
    
    sns.countplot(x='target', data=df)
    plt.title('Distribution of Target Classes')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

    corr_matrix = df.corr()
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    
    df.drop('target', axis=1).hist(bins=20, figsize=(20, 15))
    plt.suptitle('Histograms of Features')
    plt.show()

if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test, data = load_and_prepare_data()
    visualize_data(data.data, data.target, data)
