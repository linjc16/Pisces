import torch
from sklearn import metrics

def accuracy(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred (numpy.array): 
        y_true (numpy.array): [description]
    """
    return metrics.accuracy_score(y_pred=y_pred.round(), y_true=y_true)

def precision(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred (numpy.array): 
        y_true (numpy.array): [description]
    """
    return metrics.precision_score(y_pred=y_pred.round(), y_true=y_true)

def recall(y_pred, y_true):
    """
    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred ([type]): [description]
        y_true ([type]): [description]

    Returns:
        [type]: [description]
    """
    return metrics.recall_score(y_pred=y_pred.round(), y_true=y_true)

def roc_auc(y_pred, y_true):
    """
    The values of y_pred can be decimicals, within 0 and 1.
    Args:
        y_pred ([type]): [description]
        y_true ([type]): [description]

    Returns:
        [type]: [description]
    """
    return metrics.roc_auc_score(y_score=y_pred, y_true=y_true)

def pr_auc(y_pred, y_true):
    return metrics.average_precision_score(y_score=y_pred, y_true=y_true)

def f1_score(y_pred, y_true):
    return metrics.f1_score(y_pred=y_pred.round(), y_true=y_true)

