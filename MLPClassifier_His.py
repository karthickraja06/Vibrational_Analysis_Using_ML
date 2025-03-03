from sklearn.neural_network import MLPClassifier
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

# Setup arrays to store training and test accuracies
hidden_layer_sizes = [(50,), (100,), (100, 50), (100, 100)]  # Trying different network sizes
train_accuracy = np.empty(len(hidden_layer_sizes))
test_accuracy = np.empty(len(hidden_layer_sizes))

# To store loss curves for each hidden layer size
loss_curves = []

# Loop over different network configurations
for i, hls in enumerate(hidden_layer_sizes):
    # Setup neural network classifier (MLP)
    mlp = MLPClassifier(hidden_layer_sizes=hls, max_iter=1000, random_state=42)
    
    # Fit the model
    mlp.fit(X_train, y_train)
    
    # Store accuracy
    train_accuracy[i] = mlp.score(X_train, y_train)
    test_accuracy[i] = mlp.score(X_test, y_test)
    
    # Store loss curve
    loss_curves.append(mlp.loss_curve_)

# Plot accuracy comparison for different network sizes
layer_labels = [str(hls) for hls in hidden_layer_sizes]
x_range = np.arange(len(hidden_layer_sizes))

plt.figure(figsize=(12, 5))

plt.bar(x_range - 0.15, train_accuracy, width=0.3, label='Training Accuracy')
plt.bar(x_range + 0.15, test_accuracy, width=0.3, label='Testing Accuracy')
plt.xticks(x_range, layer_labels, rotation=45)
plt.legend()
plt.xlabel('Hidden Layer Sizes')
plt.ylabel('Accuracy')
plt.title('Neural Network (MLP) - VBL-VA001')
plt.show()

# Plot loss curves for each network size
plt.figure(figsize=(12, 8))
for i, hls in enumerate(hidden_layer_sizes):
    plt.plot(loss_curves[i], label=f'Hidden Layer {hls}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves for Different Hidden Layer Sizes')
plt.grid(True)
plt.show()

# Print best configuration and max test accuracy
best_index = np.argmax(test_accuracy)
print(f"Best Hidden Layer Size: {hidden_layer_sizes[best_index]}")
print(f"Max Test Accuracy: {test_accuracy[best_index]:.4f}")