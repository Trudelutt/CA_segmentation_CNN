import tensorflow as tf
from keras import backend as K
import numpy as np

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def binary_accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return K.mean(K.equal(y_true, K.round(y_pred)))


def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def dsc(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(y_true)
    predicted_positives =  K.sum(K.round(K.clip(y_pred, 0, 1)))
    return 2. * true_positives / (possible_positives + predicted_positives+ K.epsilon())


def dsc_loss(y_true, y_pred):
    return 1-dsc(y_true, y_pred)




"""def precision(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    return tp /  K.sum(y_pred_f)

def recall(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    return tp /  K.sum(y_true_f)

def accuracy(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_neg_f = 1- y_true_f
    y_pred_neg_f = 1 - y_pred_f
    tp =  K.sum(y_true_f * y_pred_f)
    fp =  K.sum(y_pred_f)- tp
    tn = K.sum(y_true_neg_f * y_pred_neg_f)
    fn =  K.sum(y_pred_neg_f)- tn
    return (tp + tn) / (tp + tn + fp + fn)

def dsc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_neg_f = 1- y_true_f
    y_pred_neg_f = 1 - y_pred_f
    tp =  K.sum(y_true_f * y_pred_f)
    fp =  K.sum(y_pred_f)- tp
    tn = K.sum(y_true_neg_f * y_pred_neg_f)
    fn =  K.sum(y_pred_neg_f)- tn
    return (2.*tp + tn) / (2.*tp + fp + fn)

def dsc_loss(y_true, y_pred):
    return 1- dsc(y_true, y_pred)"""

if __name__ == "__main__":
     var = K.variable([[0, 1, 0], [1, 0, 1]])
     var2 = K.variable([[1, 1, 1], [0, 0, 0]])
     print(K.eval(accuracy(var, var2)))
