from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load data hasil ekstraksi fitur fft
x = pd.read_csv("data/feature_VBL-VA001.csv", header=None)

# Load label
y = pd.read_csv("data/label_VBL-VA001.csv", header=None)

# Make 1D array to avoid warning
y = pd.Series.ravel(y)

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)

print("Shape of Train Data : {}".format(X_train.shape))
print("Shape of Test Data : {}".format(X_test.shape))

# Setup array to store training and test accuracies for different regularization strengths
C_values = np.logspace(-3, 3, 20)  # Test 20 values from 0.001 to 1000
train_accuracy = []
test_accuracy = []

# To store loss-like metric (LogisticRegression doesn't give true loss curve, but we can store negative log-likelihood)
loss_like_curve = []

# Loop over different regularization strengths
for C in C_values:
    # Setup logistic regression model
    logreg = LogisticRegression(C=C, max_iter=1000, random_state=42)
    
    # Fit the model
    logreg.fit(X_train, y_train)
    
    # Store accuracy
    train_accuracy.append(logreg.score(X_train, y_train))
    test_accuracy.append(logreg.score(X_test, y_test))
    
    # LogisticRegression doesn't expose a loss curve directly
    # As a proxy, we compute the negative log-likelihood (higher is worse)
    prob_train = logreg.predict_proba(X_train)
    log_likelihood = np.mean([np.log(prob_train[i, y_train[i]]) for i in range(len(y_train))])
    loss_like_curve.append(-log_likelihood)
 # Negative log-likelihood (like "loss")

# Convert to numpy arrays for easier plotting
train_accuracy = np.array(train_accuracy)
test_accuracy = np.array(test_accuracy)

# Plot accuracy vs regularization strength
plt.figure(figsize=(12, 5))

plt.plot(C_values, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(C_values, test_accuracy, label='Testing Accuracy', marker='o')
plt.xscale('log')
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Logistic Regression - VBL-VA001')
plt.grid(True)
plt.show()

# Plot "loss-like curve" vs regularization strength
plt.figure(figsize=(12, 5))

plt.plot(C_values, loss_like_curve, label='Negative Log-Likelihood (Training)', marker='o')
plt.xscale('log')
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Negative Log-Likelihood')
plt.legend()
plt.title('Logistic Regression Loss-like Curve - VBL-VA001')
plt.grid(True)
plt.show()

# Print best regularization strength and max test accuracy
best_index = np.argmax(test_accuracy)
print(f"Best Regularization Strength (C): {C_values[best_index]:.4f}")
print(f"Max Test Accuracy: {test_accuracy[best_index]:.4f}")