import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Task1 import X_train, X_test, y_train, y_test

# a)
'''
We need to implement a logistic discrimination classifier. This classifier will help us categorize songs 
based on their features into Pop or Classical using machine learning. We will use a process called 
stochastic gradient descent to train the classifier. During training, we'll track the error, experiment 
with different learning rates (how fast the model learns), and finally check how accurate the model is.
'''

# Sigmoid function to map linear combination of inputs to probability
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy loss function
def binary_cross_entropy(y_true, y_pred):
    # Clip y_pred to prevent log(0)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Accuracy function
def accuracy(y_true, y_pred):
    predictions = [1 if p > 0.5 else 0 for p in y_pred]
    return np.mean(y_true == predictions)

# Logistic Regression class using batch gradient descent
class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    # Fitting the model (train)
    def fit(self, X, y):
        num_samples, num_features = X.shape
        # Initializing weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0
        # Gradient descent
        for i in range(self.num_iterations):
            # Linear model (z = wX + b)
            z = np.dot(X, self.weights) + self.bias
            # Applying the sigmoid function to get probabilities
            y_pred = sigmoid(z)
            # Computint the gradients for weights and bias
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)
            # Updating the weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            # Storing the loss for this epoch
            loss = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

    # Predicting probabilities for new data
    def predict_prob(self, X):
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    # Predicting class labels (0 or 1) for new data
    def predict(self, X):
        probabilities = self.predict_prob(X)
        return [1 if p > 0.5 else 0 for p in probabilities]

# Plotting training error as a function of epochs and report accuracy
def plot_training_error(learning_rate, num_iterations=1000):
    # Initializing and train the Logistic Regression model
    model = LogisticRegression(learning_rate=learning_rate, num_iterations=num_iterations)
    model.fit(X_train, y_train)

    # Plotting the training error (loss) as a function of epochs
    plt.plot(range(num_iterations), model.loss_history, label=f"LR={learning_rate}")
    plt.xlabel('Epochs')
    plt.ylabel('Training Error (Loss)')
    plt.title('Training Error vs Epochs')
    plt.legend()

    # Calculating and print accuracy on the training set
    y_pred = model.predict(X_train)
    train_accuracy = accuracy(y_train, y_pred)
    print(f"Training accuracy with learning rate {learning_rate}: {train_accuracy:.4f}")

learning_rates = [0.001, 0.01, 0.1, 1] 
num_epochs = 1000

if __name__ == "__main__":
    # Plotting the training error for different learning rates
    plt.figure()
    for lr in learning_rates:
        plot_training_error(lr, num_iterations=num_epochs)
    plt.show()

# b)
'''
Test your trained logistic discrimination classifier using the test set. Report the accuracy on the test set.
Is there a significant difference between the accuracy on the training and test set? If so what might that
indicate.
'''

# Training the model with the best learning rate and report the accuracy on the test set
model = LogisticRegression(learning_rate=0.1, num_iterations=3000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
test_accuracy = accuracy(y_test, y_pred)
train_accuracy = accuracy(y_train, model.predict(X_train))
# Printing the difference between the training and test accuracy
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Accuracy difference: {np.abs(train_accuracy - test_accuracy):.4f}")

# c)
'''
[Bonus] Extract the learned parameters from your logistic regression and use them to draw the linear
line separating the data on the plot you made in question (1d). This may help you understand why your
classifier is performing well or not.
'''

# Plotting the decision boundary using the learned parameters
def plot_decision_boundary(learning_rate, num_iterations=1000):
    # Training the model
    model = LogisticRegression(learning_rate=learning_rate, num_iterations=num_iterations)
    model.fit(X_train, y_train)
    # Extracting the learned weights and bias (the parameters that define the boundary)
    w1, w2 = model.weights
    b = model.bias
    # Plotting the original data (Pop and Classical songs)
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Classical (0)', alpha=0.5)
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], facecolors='none', edgecolors='red', label='Pop (1)', alpha=0.5)
    # Plotting the decision boundary (the line that separates Pop and Classical songs)
    x_values = np.array([X_train[:, 0].min(), X_train[:, 0].max()])
    y_values = -(w1 / w2) * x_values - (b / w2)
    plt.plot(x_values, y_values, color='green', label='Decision Boundary')
    plt.xlabel('Liveness')
    plt.ylabel('Loudness')
    plt.title('Liveness vs Loudness with Decision Boundary')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Plotting the decision boundary
    plot_decision_boundary(learning_rate=0.01, num_iterations=1000)


###########################
