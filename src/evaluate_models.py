import joblib
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"{model_name} Classification Report")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}\n")

if __name__ == "__main__":
    from data_preparation import load_and_prepare_data
    X_train_scaled, X_test_scaled, y_train, y_test, data = load_and_prepare_data()

    logistic_regression = joblib.load('models/logistic_regression.pkl')
    decision_tree = joblib.load('models/decision_tree.pkl')
    knn = joblib.load('models/knn.pkl')
    naive_bayes = joblib.load('models/naive_bayes.pkl')

    evaluate_model(logistic_regression, X_test_scaled, y_test, "Logistic Regression")
    evaluate_model(decision_tree, X_test_scaled, y_test, "Decision Tree")
    evaluate_model(knn, X_test_scaled, y_test, "k-Nearest Neighbors (kNN)")
    evaluate_model(naive_bayes, X_test_scaled, y_test, "Naive Bayes")
