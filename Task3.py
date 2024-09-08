import numpy as np
from Task2 import y_pred, y_test

# a)
# Computing the confusion matrix
def confusion_matrix(y_true, y_pred):
    # Initializing the counts for TN, FP, FN, and TP
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

# Calling the function to compute the confusion matrix
c_matrix = confusion_matrix(y_test, y_pred)

print('Confusion Matrix:', '\n', c_matrix)