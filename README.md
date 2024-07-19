# Breast Cancer Classification

This project aims to evaluate and compare the performance of various machine learning algorithms in classifying breast tumors as malignant or benign using the Breast Cancer Wisconsin (Diagnostic) Dataset from the UCI Machine Learning Repository.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)s
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Introduction

Breast cancer is one of the most common cancers among women worldwide. Early detection through accurate classification of tumors can significantly improve treatment outcomes. This project utilizes several machine learning algorithms to classify breast tumors based on diagnostic features.gnostic features.

## Project Structure
```bash
.
├── notebooks
│   ├── breast_cancer_analysis.ipynb  # Comprehensive notebook for the project
├── src
│   ├── data_preparation.py       # Data preparation script
│   ├── train_models.py           # Model training script
│   ├── evaluate_models.py        # Model evaluation script
├── README.md
├── requirements.txt
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/DefineUser/breast-cancer-classification.git
cd breast-cancer-classification
```

### 2. Create and activate a virtual environment
```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

### 3. Install the required packages
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation:
Run the data preparation script to preprocess the dataset.
```bash
python src/data_preparation.py
```

### 2. Model Training:
Train the machine learning models.
```bash
python src/train_models.py
```

### 3. Model Evaluation:
Evaluate the performance of the trained models.
```bash
python src/evaluate_models.py
```

## Models
The following machine learning algorithms were implemented and optimized:

- Logistic Regression
- Decision Tree
- k-Nearest Neighbors (kNN)
- Naive Bayes

## Evaluation
The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score

Additionally, techniques to handle class imbalance, such as Random Oversampling, Random Undersampling, and SMOTE, were applied and evaluated.

## Results

The Logistic Regression model achieved the highest overall performance, with the following key metrics:

- Accuracy: 0.9825
- Precision (Weighted): 0.9825
- Recall (Weighted): 0.9825
- F1 Score (Weighted): 0.9825

For detailed results, please refer to the Jupyter notebooks in the notebooks directory.

## Contributing
Contributions are welcome! Please read the contributing guidelines before making any contributions.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## References 

1. W.N. Street, W.H. Wolberg and O.L. Mangasarian. "Nuclear feature extraction for breast tumor diagnosis." IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870, San Jose, CA, 1993. Link to dataset.
2. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research, 12:2825-2830, 2011.
3. Chawla, N. V., Bowyer, K. W., Hall, L. O., and Kegelmeyer, W. P. "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research, 16:321-357, 2002.
4. Hastie, T., Tibshirani, R., and Friedman, J. "The Elements of Statistical Learning: Data Mining, Inference, and Prediction." Springer Series in Statistics, 2009.
5. James, G., Witten, D., Hastie, T., and Tibshirani, R. "An Introduction to Statistical Learning with Applications in R." Springer, 2013.
6. Bishop, C. M. "Pattern Recognition and Machine Learning." Springer, 2006