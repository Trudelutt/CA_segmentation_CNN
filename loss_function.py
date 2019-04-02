from functools import partial

from keras import backend as K

from os.path import join
from tqdm import tqdm
import SimpleITK as sitk
from preprossesing import get_train_val_test

import numpy as np
import tensorflow as tf


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-1, -2, -3), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))

def compute_class_weights(train_data_list):
    '''
        We want to weight the the positive pixels by the ratio of negative to positive.
        Three scenarios:
            1. Equal classes. neg/pos ~ 1. Standard binary cross-entropy
            2. Many more negative examples. The network will learn to always output negative. In this way we want to
               increase the punishment for getting a positive wrong that way it will want to put positive more
            3. Many more positive examples. We weight the positive value less so that negatives have a chance.
    '''
    pos = 0.0
    neg = 0.0
    for img_name in tqdm(train_data_list):
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_name[1]))
        for slic in img:
            if not np.any(slic):
                continue
            else:
                p = np.count_nonzero(slic)
                pos += p
                neg += (slic.size - p)

    return neg/pos

def load_class_weights( label):
    class_weight_filename = join( label+'_split_lists', 'train_split_class_weights.npy')
    try:
        return np.load(class_weight_filename)
    except:
        print('Class weight file {} not found.\nComputing class weights now. This may take '
              'some time.'.format(class_weight_filename))
        train_data_list, _, _ = get_train_val_test(label)
        value = compute_class_weights(train_data_list)
        np.save(class_weight_filename,value)
        print('Finished computing class weights. This value has been saved for this training split.')
        return value


def weighted_binary_crossentropy_loss(pos_weight):
    # pos_weight: A coefficient to use on the positive examples.
    def weighted_binary_crossentropy(target, output, from_logits=False):
        """Binary crossentropy between an output tensor and a target tensor.
        # Arguments
            target: A tensor with the same shape as `output`.
            output: A tensor.
            from_logits: Whether `output` is expected to be a logits tensor.
                By default, we consider that `output`
                encodes a probability distribution.
        # Returns
            A tensor.
        """
        # Note: tf.nn.sigmoid_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # transform back to logits
            _epsilon = tf.convert_to_tensor(1e-7, output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
            output = tf.log(output / (1 - output))

        return tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                       logits=output,
                                                        pos_weight=pos_weight)
    return weighted_binary_crossentropy

def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)




def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
