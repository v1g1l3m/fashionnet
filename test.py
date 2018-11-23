# -*- coding: utf-8 -*-
import os
import sys
import pickle
import glob
import numpy as np
import functools
from keras import backend as K
from keras.models import model_from_json, load_model
from keras.utils import plot_model
# from sklearn.utils import class_weight
from keras.optimizers import *
# import matplotlib.pyplot as plt
from generator import Parallel_image_transformer, Parallel_np_arrays_reader
# from train import create_model
import logging

logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")

### GLOBALS
img_width = 224  # For VGG16
img_height = 224  # For VGG16
img_channel = 3
model_path = 'output3/'
if len(sys.argv) > 1:
    model_path = sys.argv[1]
    print(model_path)
fashion_dataset_path = 'fashion_data/'
btl_path = os.path.join(fashion_dataset_path, 'bottleneck_500')

class_names = ['Bomber', 'Flannel', 'Button-Down', 'Trunks', 'Culottes', 'Chinos', 'Jeggings', 'Parka', 'Henley', 'Jersey', 'Poncho', 'Sweatshorts', 'Cutoffs', 'Coat', 'Kimono', 'Sweatpants', 'Hoodie', 'Joggers', 'Leggings', 'Jumpsuit', 'Jeans', 'Blazer', 'Romper', 'Top', 'Jacket', 'Sweater', 'Cardigan', 'Skirt', 'Tank', 'Shorts', 'Blouse', 'Tee', 'Dress']
attr491 = [706, 598, 487, 989, 313, 312, 579, 660, 828, 440, 209, 498, 755, 166, 972, 274, 954, 41, 391, 739, 792, 882, 347, 346, 861, 915, 180, 939, 809, 85, 932, 117, 804, 157, 349, 548, 866, 485, 317, 418, 475, 535, 789, 952, 992, 125, 178, 741, 74, 634, 40, 111, 973, 128, 232, 659, 113, 860, 465, 223, 308, 385, 585, 76, 176, 235, 467, 91, 610, 434, 700, 834, 473, 888, 776, 405, 103, 258, 576, 442, 318, 334, 126, 798, 678, 105, 127, 82, 510, 582, 505, 515, 890, 118, 767, 950, 275, 657, 372, 140, 452, 296, 155, 651, 192, 528, 60, 783, 835, 478, 727, 566, 444, 965, 419, 526, 631, 734, 461, 703, 100, 234, 816, 521, 174, 614, 629, 612, 710, 315, 251, 587, 109, 686, 77, 5, 238, 265, 545, 603, 712, 847, 597, 773, 433, 927, 606, 80, 519, 407, 483, 757, 736, 469, 714, 609, 222, 291, 361, 893, 68, 746, 23, 642, 303, 397, 873, 658, 806, 540, 84, 940, 411, 354, 520, 821, 293, 210, 284, 930, 17, 43, 687, 628, 967, 150, 921, 987, 827, 977, 671, 812, 104, 393, 654, 39, 246, 476, 73, 132, 307, 119, 872, 203, 725, 277, 869, 72, 999, 273, 763, 756, 359, 389, 449, 842, 114, 532, 777, 974, 446, 799, 146, 416, 110, 474, 936, 19, 848, 224, 202, 489, 907, 669, 328, 396, 90, 283, 840, 279, 667, 183, 360, 544, 694, 696, 662, 917, 431, 302, 735, 941, 901, 708, 124, 188, 891, 468, 948, 414, 272, 619, 24, 889, 482, 929, 324, 201, 567, 971, 902, 768, 131, 327, 262, 701, 793, 569, 189, 871, 159, 154, 58, 947, 47, 943, 618, 723, 970, 683, 909, 960, 674, 27, 854, 691, 153, 330, 953, 649, 561, 666, 931, 38, 20, 480, 958, 325, 538, 788, 784, 423, 920, 36, 716, 624, 42, 18, 877, 785, 450, 25, 108, 562, 70, 218, 900, 774, 599, 116, 982, 531, 356, 30, 410, 648, 264, 802, 231, 880, 306, 139, 682, 45, 512, 797, 184, 638, 775, 309, 911, 245, 689, 336, 129, 986, 87, 937, 392, 862, 811, 852, 749, 48, 764, 15, 699, 887, 115, 321, 899, 208, 843, 653, 560, 543, 249, 988, 44, 747, 437, 815, 287, 429, 851, 99, 998, 276, 608, 415, 841, 141, 786, 14, 83, 800, 754, 11, 676, 133, 368, 697, 191, 984, 409, 413, 688, 438, 944, 897, 453, 300, 563, 237, 186, 207, 50, 574, 825, 593, 692, 751, 770, 565, 571, 817, 881, 0, 993, 695, 823, 457, 204, 722, 839, 239, 713, 681, 61, 268, 698, 227, 93, 913, 282, 781, 470, 236, 121, 337, 818, 878, 292, 616, 983, 933, 358, 969, 935, 820, 112, 601, 720, 831, 181, 956, 640, 1, 577, 620, 830, 226, 81, 745, 705, 310, 568, 760, 717, 441, 162, 335, 546, 196, 353, 380, 892, 837, 883, 212, 142]
attr_names = []
with open(os.path.join(fashion_dataset_path, 'Anno/list_attr_cloth.txt')) as f:
    next(f)
    next(f)
    for line in f:
        attr_names.append('-'.join(line.split()[:-1]))

model_name = os.path.join(model_path, 'best_model-011-0.0104-0.0288.h5')
if len(sys.argv) > 2:
    model_name = os.path.join(model_path, 'best_model.h5')
    cand = glob.glob(model_path + 'best_model*.h5')
    if sys.argv[2] == 'b':
        model_name = cand[-1]
    else:
        for n in cand:
            l = int(n.split('-')[1])
            if int(sys.argv[2]) == l:
                model_name = n
print(model_name)


# def loss(y_true, y_pred):
#     y_true_present = y_true[..., 0:1]
#     loss = y_true_present * K.mean(K.categorical_crossentropy(y_true[..., 5:], y_pred[..., 5:]), axis=-1) + \
#            y_true_present * K.mean(K.square(y_pred[..., 1:5] - y_true[..., 1:5]), axis=-1) + \
#            (1 - y_true_present) * K.mean(K.square(y_true_present - y_pred[..., 0:1]), axis=-1)
#     return loss

# import keras.losses
# keras.losses.loss_cross = loss

# model = load_model(model_name)
# model.save_weights(os.path.join(model_path, 'best_weights.hdf5'))


batch_size = 64
val_steps = 11407//batch_size
count = 1

model = load_model(model_name)
model.save_weights(os.path.join(model_path, 'best_weights.hdf5'))

# with open('results.txt', "a") as w:
#     for model_name in cand[14:]:
        # model = load_model(model_name)
# with Parallel_image_transformer('/media/star/3C4C65AA4C65601E/dev/deepfashion/fashion_data/all.txt', (batch_size, 224, 224, 3)) as val_gen:
with Parallel_image_transformer('/media/star/3C4C65AA4C65601E/dev/deepfashion/fashion_data/validation_95-ac.txt', (batch_size, 224, 224, 3)) as val_gen:
# with Parallel_np_arrays_reader(os.path.join(btl_path, 'btl_validation_npz.txt'), ['attr_cls'], 5) as val_gen:
    X, y = next(val_gen)
    Ya = y[0]
    Yc = y[1]
    y_pred = model.predict(X, batch_size=batch_size)
    classes = y_pred[1]
    attrs = y_pred[0]
    for i in range(val_steps-1):
        X, y = next(val_gen)
        Ya = np.concatenate([Ya, y[0]])
        Yc = np.concatenate([Yc, y[1]])
        y_pred = model.predict(X, batch_size=batch_size)
        attrs = np.concatenate([attrs, y_pred[0]])
        classes = np.concatenate([classes, y_pred[1]])


cls_TP = dict((x, 0) for x in range(len(class_names)))
cls_FN = dict((x, 0) for x in range(len(class_names)))
cls_FP = dict((x, 0) for x in range(len(class_names)))
for i, y_act in enumerate(Yc):
    t = np.argmax(classes[i])
    y = np.argmax(y_act)
    if t != y:
        cls_FN[y] += 1
        cls_FP[t] += 1
    else:
        cls_TP[t] += 1
class_precision, class_recall, class_f1 = np.zeros((len(class_names))), np.zeros((len(class_names))), np.zeros((len(class_names)))
for x in range(len(class_names)):
    if cls_TP[x] > 0:
        class_precision[x] = cls_TP[x]/(cls_TP[x]+cls_FP[x])
        class_recall[x] = cls_TP[x]/(cls_TP[x]+cls_FN[x])
        class_f1[x] = 2*class_precision[x]*class_recall[x]/(class_precision[x]+class_recall[x])

# thres = list(np.linspace(0.15, 0.25, 11))+[1,3,5,7,10]
# f1_list = []
# for th in thres:
#     attr_TP = dict((x, 0) for x in range(491))
#     attr_FN = dict((x, 0) for x in range(491))
#     attr_FP = dict((x, 0) for x in range(491))
#     for i, attr in enumerate(Ya):
#         if th > 0.51:
#             topN = []
#             ypred = np.copy(attrs[i])
#             for k in range(th):
#                 topN.append(np.argmax(ypred))
#             ypred[topN[-1]] = 0.
#             t = set(topN)
#         else:
#             pred = [x[0] for x in np.argwhere(attrs[i] > th)]
#             t = set(pred)
#         y = set([x[0] for x in np.argwhere(attr == 1)])
#         for x in (t&y):
#             attr_TP[x] += 1
#         for x in (y - t):
#             attr_FN[x] += 1
#         for x in (t - y):
#             attr_FP[x] += 1
#     attr_precision, attr_recall, attr_f1 = np.zeros((491)), np.zeros((491)), np.zeros((491))
#     for x in range(491):
#         if attr_TP[x] > 0:
#             attr_precision[x] = attr_TP[x]/(attr_TP[x]+attr_FP[x])
#             attr_recall[x] = attr_TP[x]/(attr_TP[x]+attr_FN[x])
#             attr_f1[x] = 2*attr_precision[x]*attr_recall[x]/(attr_precision[x]+attr_recall[x])
#     f1_list.append((np.mean([x for x in attr_f1]), np.mean([x for x in attr_precision]), np.mean([x for x in attr_recall])))
# best = sorted([(f1_list[i][0], f1_list[i][1], f1_list[i][2], thres[i]) for i in range(len(thres))], reverse=True)[0]
# print(best)
# w.write('{};{};{};{};{}\n'.format(model_name, best[0],best[1],best[2],best[3]))
th = 0.21
attr_TP = dict((x, 0) for x in range(491))
attr_FN = dict((x, 0) for x in range(491))
attr_FP = dict((x, 0) for x in range(491))
for i, attr in enumerate(Ya):
    if th > 0.51:
        topN = []
        ypred = attrs[i, :]
        for k in range(th):
            topN.append(np.argmax(ypred))
        ypred[topN[-1]] = 0.
        t = set(topN)
    else:
        pred = [x[0] for x in np.argwhere(attrs[i] > th)]
        t = set(pred)
    y = set([x[0] for x in np.argwhere(attr == 1)])
    for x in (t&y):
        attr_TP[x] += 1
    for x in (y - t):
        attr_FN[x] += 1
    for x in (t - y):
        attr_FP[x] += 1
attr_precision, attr_recall, attr_f1 = np.zeros((491)), np.zeros((491)), np.zeros((491))
for x in range(491):
    if attr_TP[x] > 0:
        attr_precision[x] = attr_TP[x]/(attr_TP[x]+attr_FP[x])
        attr_recall[x] = attr_TP[x]/(attr_TP[x]+attr_FN[x])
        attr_f1[x] = 2*attr_precision[x]*attr_recall[x]/(attr_precision[x]+attr_recall[x])



for i in range(491):
    print('{}%, {}% of attribute {}: {}'.format('%.0f'%(attr_precision[i]*100),'%.0f'%(attr_recall[i]*100), attr_names[attr491[i]], attr_TP[i]+attr_FN[i]))
print('--------------------------------------------------------')
for i, cls in enumerate(class_names):
    print('{}%, {}% of class {}: {}'.format('%.0f'%(class_precision[i]*100),'%.0f'%(class_recall[i]*100), cls, cls_TP[i]+cls_FN[i]))

print(count,'Class precision: ', np.mean([x for x in class_precision if x > 0]))
print(count,'Class recall: ', np.mean([x for x in class_recall if x > 0]))
print(count,' Class F1-score: ', np.mean([x for x in class_f1 if x > 0]))
print(count,'Attribute precision: ', np.mean([x for x in attr_precision]))
print(count,'Attribute recall: ', np.mean([x for x in attr_recall]))
print(count, 'Attribute F1-score: ', np.mean([x for x in attr_f1]))
count += 1
