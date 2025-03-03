# Script to train VBL-VA001

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

def load_data(feature_path, label_path):
    """Load feature and label data from CSV files."""
    x = pd.read_csv(feature_path, header=None)
    y = pd.read_csv(label_path, header=None)
    y = y.values.ravel()  # Convert to 1D array to avoid warnings
    return x, y

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train and evaluate the model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, matrix

def plot_confusion_matrix(matrix, title):
    """Plot the confusion matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(matrix))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def main():
    # Load data
    feature_path = "data/feature_VBL-VA001.csv"
    label_path = "data/label_VBL-VA001.csv"
    X, y = load_data(feature_path, label_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # Train and evaluate Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb_accuracy, gnb_report, gnb_matrix = train_and_evaluate_model(gnb, X_train, X_test, y_train, y_test)
    print("Gaussian Naive Bayes Accuracy:", gnb_accuracy)
    print("Gaussian Naive Bayes Classification Report:\n", gnb_report)
    plot_confusion_matrix(gnb_matrix, "Gaussian Naive Bayes Confusion Matrix")

    # Train and evaluate Support Vector Classifier
    svc = SVC()
    svc_accuracy, svc_report, svc_matrix = train_and_evaluate_model(svc, X_train, X_test, y_train, y_test)
    print("Support Vector Classifier Accuracy:", svc_accuracy)
    print("Support Vector Classifier Classification Report:\n", svc_report)
    plot_confusion_matrix(svc_matrix, "Support Vector Classifier Confusion Matrix")

    # Train and evaluate K-Nearest Neighbors
    knn = KNeighborsClassifier()
    knn_accuracy, knn_report, knn_matrix = train_and_evaluate_model(knn, X_train, X_test, y_train, y_test)
    print("K-Nearest Neighbors Accuracy:", knn_accuracy)
    print("K-Nearest Neighbors Classification Report:\n", knn_report)
    plot_confusion_matrix(knn_matrix, "K-Nearest Neighbors Confusion Matrix")

if __name__ == "__main__":
    main()