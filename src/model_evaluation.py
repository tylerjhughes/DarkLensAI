import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of a trained model on a test set.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    results = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall'],
                            'Score': [accuracy, precision, recall]})
    
    return results