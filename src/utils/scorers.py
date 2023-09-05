######
# This script contains custom scoring functions that accept
# a different threshold for binary classification.
######
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def f1_w_threshold(y_true, y_pred_prob, threshold):
    y_pred = (y_pred_prob >= threshold).astype(int)
    return f1_score(y_true, y_pred)

def acc_w_threshold(y_true, y_pred_prob, threshold):
    y_pred = (y_pred_prob >= threshold).astype(int)
    return accuracy_score(y_true, y_pred)

def precision_w_threshold(y_true, y_pred_prob, threshold):
    y_pred = (y_pred_prob >= threshold).astype(int)
    return precision_score(y_true, y_pred)

def recall_w_threshold(y_true, y_pred_prob, threshold):
    y_pred = (y_pred_prob >= threshold).astype(int)
    return recall_score(y_true, y_pred)
