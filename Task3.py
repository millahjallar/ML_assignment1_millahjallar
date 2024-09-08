import matplotlib.pyplot as plt
import numpy as np
from Task2 import y_pred, y_test

# Manually compute the confusion matrix
def confusion_matrix_manual(y_true, y_pred):
    # Initialize the counts for TN, FP, FN, and TP
    TN = FP = FN = TP = 0
    
    for true, pred in zip(y_true, y_pred):
        if true == 0 and pred == 0:
            TN += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and pred == 0:
            FN += 1
        elif true == 1 and pred == 1:
            TP += 1
    
    return np.array([[TN, FP], [FN, TP]])

# Assuming y_test and y_pred exist in your environment

# Call the function to compute the confusion matrix
cm_manual = confusion_matrix_manual(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm_manual)

