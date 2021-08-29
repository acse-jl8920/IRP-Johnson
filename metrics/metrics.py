#coding=utf-8
import tensorflow as tf
from keras.utils.generic_utils import *

__all__ = ['iou_score', 'jaccard_score', 'f1_score', 'f2_score',
            'dice_score', 'get_f_score', 'get_iou_score', 'get_jaccard_score']

SMOOTH = 1.

# ============================ Jaccard/IoU score ============================
def iou_score(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True, threshold=None):
    '''
    参数：
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), 
        if ``None`` prediction prediction will not be round
    返回：
        IoU/Jaccard score in range [0, 1]
    '''
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]
        
    if threshold is not None:
        pr = tf.greater(pr, threshold)
        pr = tf.cast(pr, dtype=tf.float32)

    intersection = tf.reduce_sum(gt * pr, axis=axes)
    union = tf.reduce_sum(gt + pr, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)

    # mean per image
    if per_image:
        iou = tf.reduce_mean(iou, axis=0)

    # weighted mean per class
    iou = tf.reduce_mean(iou * class_weights)

    return iou

# 计算IOU得分
def get_iou_score(class_weights=1., smooth=SMOOTH, per_image=True, threshold=None):
    def score(gt, pr):
        return iou_score(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image, threshold=threshold)
    return score

jaccard_score = iou_score
get_jaccard_score = get_iou_score

get_custom_objects().update({'iou_score': iou_score,
                            'jaccard_score': jaccard_score})


# ============================== F/Dice - score ==============================

def f_score(gt, pr, class_weights=1, beta=1, smooth=SMOOTH, per_image=True, threshold=None):


    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]
        
    if threshold is not None:
        pr = tf.greater(pr, threshold)
        pr = tf.cast(pr, dtype=tf.float32)

    tp = tf.reduce_sum(gt * pr, axis=axes)
    fp = tf.reduce_sum(pr, axis=axes) - tp
    fn = tf.reduce_sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)

    # mean per image
    if per_image:
        score = tf.reduce_mean(score, axis=0)

    # weighted mean per class
    score = tf.reduce_mean(score * class_weights)

    return score

def get_f_score(class_weights=1, beta=1, smooth=SMOOTH, per_image=True, threshold=None):
    '''
    参数:
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        beta: f-score coefficient
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction prediction will not be round
    返回:
        ``callable``: F-score
    '''

    def score(gt, pr):
        return f_score(gt, pr, class_weights=class_weights, beta=beta, smooth=smooth, per_image=per_image, threshold=threshold)

    return score

f1_score = get_f_score(beta=1)
f2_score = get_f_score(beta=2)
dice_score = f1_score

# Update custom objects
get_custom_objects().update({
    'f1_score': f1_score,
    'f2_score': f2_score,
    'dice_score': dice_score,
})
