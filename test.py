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
from generator import Parallel_image_transformer2, Parallel_image_transformer3, Parallel_np_arrays_reader2, \
    Parallel_np_arrays_reader1
from utils import init_globals, bb_intersection_over_union, draw_rect, get_validation_data
# from train import create_model
import logging

logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")

### GLOBALS
img_width = 224  # For VGG16
img_height = 224  # For VGG16
img_channel = 3
model_path = 'output/'
if len(sys.argv) > 1:
    model_path = sys.argv[1]
    print(model_path)
fashion_dataset_path = 'fashion_data/'
btl_path = os.path.join(fashion_dataset_path, 'bottleneck226_350')

global class_names, input_shape, attr_names, class35, attr200
class_names, input_shape, attr_names = init_globals(fashion_dataset_path)
class35 = ['Blazer', 'Top', 'Dress', 'Chinos', 'Jersey', 'Cutoffs', 'Kimono', 'Cardigan', 'Jeggings', 'Button-Down',
           'Romper', 'Skirt', 'Joggers', 'Tee', 'Turtleneck', 'Culottes', 'Coat', 'Henley', 'Jeans', 'Hoodie', 'Blouse',
           'Tank', 'Shorts', 'Bomber', 'Jacket', 'Parka', 'Sweatpants', 'Leggings', 'Flannel', 'Sweatshorts',
           'Jumpsuit', 'Poncho', 'Trunks', 'Sweater', 'Robe']
attr200 = [730, 365, 513, 495, 836, 596, 822, 254, 884, 142, 212, 883, 837, 892, 380, 353, 196, 546, 335, 162, 441, 717,
           760, 568, 310, 705, 745, 81, 226, 830, 620, 577, 1, 640, 956, 181, 831, 720, 601, 112, 820, 935, 969, 358,
           933, 983, 616, 292, 878, 818, 337, 121, 236, 470, 781, 282, 913, 93, 227, 698, 268, 61, 681, 713, 239, 839,
           722, 204, 457, 823, 695, 993, 0, 881, 817, 571, 565, 770, 751, 692, 593, 825, 574, 50, 207, 186, 237, 563,
           300, 453, 897, 944, 438, 688, 413, 409, 984, 191, 697, 368, 133, 676, 11, 754, 800, 83, 14, 786, 141, 841,
           415, 608, 276, 998, 99, 851, 429, 287, 815, 437, 747, 44, 988, 249, 543, 560, 653, 843, 208, 899, 321, 115,
           887, 699, 15, 764, 48, 749, 852, 811, 862, 392, 937, 87, 986, 129, 336, 689, 245, 911, 309, 775, 638, 184,
           797, 512, 45, 682, 139, 306, 880, 231, 802, 264, 648, 410, 30, 356, 531, 982, 116, 599, 774, 900, 218, 70,
           562, 108, 25, 450, 785, 877, 18, 42, 624, 716, 36, 920, 423, 784, 788, 538, 325, 958, 480, 20, 38, 931, 666,
           561]
attr350 = [483, 757, 736, 469, 714, 609, 222, 291, 361, 893, 68, 746, 23, 642, 303, 397, 873, 658, 806, 540, 84, 940, 411, 354, 520, 821, 293, 210, 284, 930, 17, 43, 687, 628, 967, 150, 921, 987, 827, 977, 671, 812, 104, 393, 654, 39, 246, 476, 73, 132, 307, 119, 872, 203, 725, 277, 869, 72, 999, 273, 763, 756, 359, 389, 449, 842, 114, 532, 777, 974, 446, 799, 146, 416, 110, 474, 936, 19, 848, 224, 202, 489, 907, 669, 328, 396, 90, 283, 840, 279, 667, 183, 360, 544, 694, 696, 662, 917, 431, 302, 735, 941, 901, 708, 124, 188, 891, 468, 948, 414, 272, 619, 24, 889, 482, 929, 324, 201, 567, 971, 902, 768, 131, 327, 262, 701, 793, 569, 189, 871, 159, 154, 58, 947, 47, 943, 618, 723, 970, 683, 909, 960, 674, 27, 854, 691, 153, 330, 953, 649, 561, 666, 931, 38, 20, 480, 958, 325, 538, 788, 784, 423, 920, 36, 716, 624, 42, 18, 877, 785, 450, 25, 108, 562, 70, 218, 900, 774, 599, 116, 982, 531, 356, 30, 410, 648, 264, 802, 231, 880, 306, 139, 682, 45, 512, 797, 184, 638, 775, 309, 911, 245, 689, 336, 129, 986, 87, 937, 392, 862, 811, 852, 749, 48, 764, 15, 699, 887, 115, 321, 899, 208, 843, 653, 560, 543, 249, 988, 44, 747, 437, 815, 287, 429, 851, 99, 998, 276, 608, 415, 841, 141, 786, 14, 83, 800, 754, 11, 676, 133, 368, 697, 191, 984, 409, 413, 688, 438, 944, 897, 453, 300, 563, 237, 186, 207, 50, 574, 825, 593, 692, 751, 770, 565, 571, 817, 881, 0, 993, 695, 823, 457, 204, 722, 839, 239, 713, 681, 61, 268, 698, 227, 93, 913, 282, 781, 470, 236, 121, 337, 818, 878, 292, 616, 983, 933, 358, 969, 935, 820, 112, 601, 720, 831, 181, 956, 640, 1, 577, 620, 830, 226, 81, 745, 705, 310, 568, 760, 717, 441, 162, 335, 546, 196, 353, 380, 892, 837, 883, 212, 142, 884, 254, 822, 596, 836, 495, 513, 365, 730]


model_name = os.path.join(model_path, 'final_model.h5')
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


def loss(y_true, y_pred):
    y_true_present = y_true[..., 0:1]
    loss = y_true_present * K.mean(K.categorical_crossentropy(y_true[..., 5:], y_pred[..., 5:]), axis=-1) + \
           y_true_present * K.mean(K.square(y_pred[..., 1:5] - y_true[..., 1:5]), axis=-1) + \
           (1 - y_true_present) * K.mean(K.square(y_true_present - y_pred[..., 0:1]), axis=-1)
    return loss


import keras.losses

keras.losses.custom_loss = loss
keras.losses.loss_cross = loss
keras.losses.loss_mse = loss
keras.losses.loss_mae = loss
keras.losses.loss_cat_cross = loss

model = load_model(model_name)
model.save_weights(os.path.join(model_path, 'best_weights.hdf5'))
# exit(0)

# temp = get_validation_data(os.path.join(btl_path, 'btl_validation_npz.txt'))
# X = np.array(temp['btl'])
# Yb = np.array(temp['bbox'])
# Ya = np.array(temp['attr'])
# bboxes, attrs = model.predict(X, 48, verbose=1)
# y_pred = model.predict(X, verbose=1)
# pc = y_pred[:, 0]
# bboxes = y_pred[:, 1:5]
# attrs = y_pred[:, 5:]
# with open(os.path.join(model_path, 'out.txt')) as r:
#     with open(os.path.join(model_path, 'out2.txt'), "w") as wr:
#         for l1 in r:
#             l2 = r.readline()
#             l1 = l1.rstrip().split('-')
#             l2 = l2.rstrip().split('-')
#             wr.write('{} - {}, {}\n'.format(l1[0], l1[1], l2[1]))
# with open(os.path.join(model_path, 'out.txt'), "a") as wr:
#     for ii in range(int(sys.argv[2]), int(cand[-1].split('-')[1])+1):   
#         for n in cand:
#             l = int(n.split('-')[1])
#             if ii == l:
#                 model_name = n
#         model = load_model(model_name)

batch_size = 226
# with Parallel_image_transformer2('/media/star/3C4C65AA4C65601E/dev/deepfashion/fashion_data/validation_95-5.txt', (batch_size, 224, 224, 3), (batch_size, 1004)) as val_gen:
with Parallel_np_arrays_reader1(os.path.join(btl_path, 'btl_validation_npz.txt'), ['bbox', 'attr', 'cls'], 10) as val_gen:
    X, y = next(val_gen)
    Yb = y[0]
    Ya = y[1]
    Yc = y[2]
    # Ypc = y[:, 0].reshape((batch_size, 1))
    bboxes, attrs,  classes = model.predict(X, batch_size=batch_size)
    # pc = y_pred[:, 0].reshape((batch_size, 1))
    # bboxes = y_pred[:, 1:5]
    # attrs = y_pred[:, 5:]
    for i in range(78):
        X, y = next(val_gen)
        Yb = np.concatenate([Yb, y[0]])
        Ya = np.concatenate([Ya, y[1]])
        Yc = np.concatenate([Yc, y[2]])
        # Ypc = np.concatenate([Ypc, y[:, 0].reshape((batch_size, 1))])
        y_pred_b, y_pred_a, y_pred_c = model.predict(X, batch_size=batch_size)
        bboxes = np.concatenate([bboxes, y_pred_b])
        attrs = np.concatenate([attrs, y_pred_a])
        classes = np.concatenate([classes, y_pred_c])
        # pc = np.concatenate([pc, y_pred[:, 0].reshape((batch_size, 1))])

# pc_er = 0
# pc_wrong_pred = 0
# for i in range(len(Ypc)):
#     if Ypc[i] - 1 < 1e-14 and pc[i] < 0.5:
#         pc_er += 1
#     if Ypc[i] < 1e-14 and pc[i] > 0.5:
#         pc_er += 1
#         pc_wrong_pred += 1

# with open(os.path.join(btl_path, 'attr_data_train.pkl'), 'rb') as f:
# train_labels_attr = pickle.load(f)
# with open(os.path.join(btl_path, 'class_data_train.pkl'), 'rb') as f:
# train_labels_class = pickle.load(f)

bbox_iou = []
for i, bb in enumerate(Yb):
    # if Ypc[i] < 1e-14:
    #     continue
    w, h = 224, 224
    bb_act = [bb[0] * w, bb[1] * h, bb[2] * w, bb[3] * h]
    bb_act = [bb_act[0] - bb_act[2] / 2, bb_act[1] - bb_act[3] / 2, bb_act[2], bb_act[3]]
    bb_pred = [bboxes[i][0] * w, bboxes[i][1] * h, bboxes[i][2] * w, bboxes[i][3] * h]
    bb_pred = [bb_pred[0] - bb_pred[2] / 2, bb_pred[1] - bb_pred[3] / 2, bb_pred[2], bb_pred[3]]
    bbox_iou.append(bb_intersection_over_union(bb_act, bb_pred))

    # bbox_iou.append(bbwh[4])
    # img = np.zeros((int(w),int(h),3), dtype=np.int8)
    # ax = plt.subplot()
    # draw_rect(ax, img, bb_act, edgecolor='green')
    # draw_rect(ax, img, bb_pred, edgecolor='red')
    # plt.show()

class_er = dict((x, 0) for x in range(len(class_names)))
class_total = dict((x, 0) for x in range(len(class_names)))
for i, cls in enumerate(Yc):
    class_total[np.argmax(cls)] += 1
    pred = classes[i]
    test = np.max(cls)
    if test < 1e-7 and np.max(pred) >= 0.5:
        # class_wrong_pred[np.argmax(pred)] += 1
        continue
    if np.argmax(cls) != np.argmax(pred):
        class_er[np.argmax(cls)] += 1
        # class_wrong_pred[np.argmax(pred)] += 1

attr_er = dict((x, 0) for x in range(350))
attr_total = dict((x, 0) for x in range(350))
for i, attr in enumerate(Ya):
    # if Ypc[i] < 1e-14:
    #     continue
    # pred = set([x[0] for x in np.argwhere(attrs[i] >= 0.5)])
    top5 = []
    for k in range(5):
        top5.append(np.argmax(attrs[i]))
        attrs[i, top5[-1]] = 0
    act = set([x[0] for x in np.argwhere(attr == 1)])
    for j in act:
        attr_total[j] += 1
        if j not in top5:
            attr_er[j] += 1
cls_acc = np.array([(class_total[x] - class_er[x]) / class_total[x] for x in range(len(class_names)) if class_total[x] > 0])
class_total_wrong_pred = np.sum([x for x in class_er.values()])
class_total_pred = np.sum([x for x in class_total.values()])
attr_total_wrong_pred = sum([x for x in attr_er.values()])
attr_total_pred = sum([x for x in attr_total.values()])
attr_sorted = sorted([((attr_total[i] - attr_er[i]) * 100 / attr_total[i],
                       attr_names[attr350[i]],
                       attr_total[i],
                       attr_er[i])
                      for i in range(350) if attr_total[i] > 0], key=lambda x: x[2])

with open(os.path.join(model_path, 'test_results.txt'), 'w') as f:
    # f.write('Test samples number: {}\n'.format(len(X)))
    f.write('Total wrong predictions for class: {}\n'.format(class_total_wrong_pred))
    f.write('Total wrong predictions for attributes: {}\n'.format(attr_total_wrong_pred))
    f.write('Class average prediction accuracy: {}\n'.format(np.mean(cls_acc)*100))
    f.write('Class summary prediction accuracy: {}\n'.format(((class_total_pred-class_total_wrong_pred)*100)/class_total_pred))
    f.write('Attribute average prediction accuracy: {}\n'.format(np.mean([x[0] for x in attr_sorted])))
    f.write('Attribute summary prediction accuracy: {}\n'.format(((attr_total_pred-attr_total_wrong_pred)*100)/attr_total_pred))
    # f.write('bbox iou average: {}\n'.format(np.mean(bbox_iou))) 
    j = 0
    for i, cls in enumerate(class_names):
        if class_total[i] == 0:
            continue
        print('{}% Total of class {}: {}, missed: {}'.format('%.2f'%(cls_acc[j]*100), cls, class_total[i], class_er[i]))
        f.write('{}% Total of class {}: {}, missed: {}\n'.format('%.2f'%(cls_acc[j]*100), cls, class_total[i], class_er[i]))
        j += 1
    print('--------------------------------------------------------')
    f.write('--------------------------------------------------------\n')
    j = 0
    for i in range(350):
        if attr_total[i] == 0:
            continue
        print('{}% Total of attribute {}: {}, missed: {}'.format('%.2f' % (attr_sorted[j][0]), attr_sorted[j][1],
                                                                 attr_sorted[j][2], attr_sorted[j][3]))
        f.write('{}% Total of attribute {}: {}, missed: {}\n'.format('%.2f' % (attr_sorted[j][0]), attr_sorted[j][1],
                                                                     attr_sorted[j][2], attr_sorted[j][3]))
        j += 1
# print('Test samples number: ', len(Yb))
print('Total wrong predictions for class: ', class_total_wrong_pred)
print('Total wrong predictions for attributes: ', attr_total_wrong_pred)
print('Class average prediction accuracy: ', np.mean(cls_acc)*100)
print('Class summary prediction accuracy: ', ((class_total_pred-class_total_wrong_pred)*100)/class_total_pred)
print('Attribute average prediction accuracy: ', np.mean([x[0] for x in attr_sorted]))
print('Attribute summary prediction accuracy: ', ((attr_total_pred-attr_total_wrong_pred)*100)/attr_total_pred)
print('bbox iou average: ', np.mean(bbox_iou))

        # print(ii)
        # wr.write(str(ii)+' - {}, {}\n'.format( ((class_total_pred-class_total_wrong_pred)*100)/class_total_pred, ((attr_total_pred-attr_total_wrong_pred)*100)/attr_total_pred ))
        # print('Average attribute accuracy: ', 100*np.mean([attr_er[i]/attr_total[i] for i in range(1000)]))
        # print ('pc prediction accuracy: ', (len(Ypc)-pc_er)*100/len(Ypc))
        # print ('pc wrong predictions: ', (pc_wrong_pred)*100/pc_er)
        # print ('pc missed predictions: ', (pc_missed_pred)/len(Ypc))
        # print(model.summary())
