from data import *
from functools import partial
from keras import backend as K
import tensorflow as tf
import matplotlib as plt
import math
from sklearn.utils.extmath import cartesian


def dice_coefficient(y_true, y_pred, smooth=2.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_accuracy(y_true, y_pred, smooth=0.):
    return dice_coefficient(y_true, y_pred, smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

def hybrid_loss(y_true,y_pred):
    return -K.binary_crossentropy(y_true,y_pred)+3*dice_coefficient_loss(y_true, y_pred)

def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
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

ALPHA = 0.8
GAMMA = 2

def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1 - BCE_EXP), gamma) * BCE)

    return focal_loss


def weighted_cross_entropy(beta):
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=beta)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)

    return loss


def dice_loss(y_true, y_pred, smooth=50):
    # y_true_flatten = tf.keras.layers.Flatten()
    # y_true_f = y_true_flatten(y_true)
    y_true_f = K.flatten(y_true)
    # y_pred_flatten = tf.keras.layers.Flatten()
    # y_pred_f = y_pred_flatten(y_pred)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    val = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    return 1 - val


def combo_loss(y_true, y_pred,alpha=0.8, beta=0.4):
    return alpha*tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=beta)+((1-alpha)*dice_coefficient_loss(y_true, y_pred))


def dice_coef(img, img2):
    if img.shape != img2.shape:
        raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
    else:
        intersection = np.logical_and(img, img2)
        value = (2. * intersection.sum()) / (img.sum() + img2.sum())
    return value

def calc_hausdorff(m1, m2):
    m1 = sitk.GetImageFromArray(m1, isVector=False)
    m2 = sitk.GetImageFromArray(m2, isVector=False)
    hausdorff = sitk.HausdorffDistanceImageFilter()
    hausdorff.Execute(m1, m2)
    return hausdorff.GetHausdorffDistance()

def calculate_statistics(imagepath_1,imagepath_2):
    all_masks1 = os.listdir(imagepath_1)
    all_masks2 = os.listdir(imagepath_2)
    ''' 
    all_masks2 = []
    for i in all_masks1:
        all_masks2.append('label'+i[5:len(i)])
    '''
    hd=[]
    dice =[]
    for mask1, mask2 in zip(all_masks1,all_masks2):
        m1 = load_grayscale_image_VTK(os.path.join(imagepath_1, mask1))
        #m1 = cv2.rotate(m1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #plt.imshow(Image.fromarray(m1[:,:]*255), cmap=plt.cm.bone)
        #plt.show()
        m2 = load_grayscale_image_VTK(os.path.join(imagepath_2, mask2))
        #m2 = cv2.rotate(m2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #plt.imshow(Image.fromarray(m2[:,:,0]*255), cmap=plt.cm.bone)
        #plt.show()
        #m1 = cv2.resize(m1[:, :], (512, 512), interpolation=cv2.INTER_NEAREST)
        #m1 = cv2.medianBlur(m1, 5)
        dice.append(dice_coef(m1, m2[:,:,0]))
        hd.append(calc_hausdorff(m1,m2[:,:,0]))


    hd=np.array(hd)
    dice = np.array(dice)
    stat_hd = [np.mean(hd),np.std(hd), np.max(hd),np.min(hd)]
    stat_dice = [np.mean(dice), np.std(dice), np.max(dice), np.min(dice)]
    sample = open('metrics.txt', 'w')
    print('Hausdorff distance: {} Dice_coefficent: {}'.format(stat_hd, stat_dice),file=sample)
    sample.close()
    return
