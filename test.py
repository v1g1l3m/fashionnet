# -*- coding: utf-8 -*-
import os
import sys
import pickle
import numpy as np
from keras.models import model_from_json, load_model
from keras.utils import plot_model
from sklearn.utils import class_weight
from keras.optimizers import *
import matplotlib.pyplot as plt
from utils import init_globals, bb_intersection_over_union, draw_rect
import logging
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")

### GLOBALS
img_width = 224             # For VGG16
img_height = 224            # For VGG16
img_channel = 3
model_path = 'output5/'
if len(sys.argv) > 1:
    model_path = sys.argv[1]
    print(model_path)
fashion_dataset_path='../Data/fashion_data/'
btl_path = 'E:\\ML\\bottleneck_df'

class_names, _, attr_names = init_globals()
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

model_name = 'final_model.h5'    
if len(sys.argv) > 2:
    if (sys.argv[2] == 'b'):
        model_name = 'best-weights.hdf5'  
model = load_model(os.path.join(model_path, model_name))
# plot_model(model, to_file=os.path.join(model_path, 'model.png'), show_shapes=True, show_layer_names=False)
# with open('output/bottleneck_fc_model.json') as f:
#     model = model_from_json(f.read())
# model.load_weights('output/bottleneck_fc_model.h5')
# model.compile(optimizer=RMSprop(lr=1e-4),loss={'predictions_bbox':'mse', 'predictions_attr': 'binary_crossentropy','predictions_class': 'categorical_crossentropy'}, metrics=['accuracy'])

X = np.load(os.path.join(fashion_dataset_path, 'btl_test10.npz'))
bboxes, attrs, classes = model.predict(X['btl'])
with open(os.path.join(btl_path, 'attr_data_train.pkl'), 'rb') as f:
    train_labels_attr = pickle.load(f)
with open(os.path.join(btl_path, 'class_data_train.pkl'), 'rb') as f:
    train_labels_class = pickle.load(f)
cls_weight = class_weight.compute_class_weight('balanced', class35, train_labels_class)
attr_weight = class_weight.compute_class_weight('balanced', attr200, train_labels_attr)
out_weights = [[1.,1.,1.,1.], attr_weight, cls_weight]

class_er = dict((x, 0) for x in range(len(class35)))
class_wrong_pred = dict((x, 0) for x in range(len(class35)))
class_total = dict((x, 0) for x in range(len(class35)))
attr_er = dict((x, 0) for x in range(len(attr200)))
attr_wrong_pred = dict((x, 0) for x in range(len(attr200)))
attr_total = dict((x, 0) for x in range(len(attr200)))
bbox_iou = []
for i, bbwh in enumerate(X['bbwh']):
    w, h = bbwh[4], bbwh[5]
    bb_act = [bbwh[0]*w, bbwh[1]*h, bbwh[2]*w, bbwh[3]*h]
    bb_pred = [bboxes[i][0]*w, bboxes[i][1]*h, bboxes[i][2]*w, bboxes[i][3]*h]
    bbox_iou.append(bb_intersection_over_union(bb_act, bb_pred))
    # img = np.zeros((int(w),int(h),3), dtype=np.int8)
    # ax = plt.subplot()
    # draw_rect(ax, img, bb_act, edgecolor='green')
    # draw_rect(ax, img, bb_pred, edgecolor='red')
    # plt.show()
for i, cls in enumerate(X['cls']):
    class_total[np.argmax(cls)] += 1
    pred = classes[i]
    test = np.max(cls)
    if test < 1e-7 and np.max(pred) >= 0.5:
        class_wrong_pred[np.argmax(pred)] += 1
        continue
    if np.argmax(cls) != np.argmax(pred):
        class_er[np.argmax(cls)] += 1
        class_wrong_pred[np.argmax(pred)] += 1

for i, attr in enumerate(X['attr']):
    pred = set([x[0] for x in np.argwhere(attrs[i] >= 0.5)])
    act = set([x[0] for x in np.argwhere(attr == 1)])
    for j in act:
        attr_total[j] += 1
    missed = act - pred
    for j in missed:
        attr_er[j] += 1
    wrong = pred - act
    for j in wrong:
        attr_wrong_pred[j] += 1
cls_acc = np.array([(class_total[x] - class_er[x])/class_total[x] for x in range(len(class35))])
class_total_wrong_pred = np.sum([x for x in class_wrong_pred.values()])
attr_acc = np.array([(attr_total[x] - attr_er[x])/attr_total[x] for x in range(len(attr200))])
attr_total_wrong_pred = np.sum([x for x in attr_wrong_pred.values()])

with open(os.path.join(model_path, 'test_results.txt'), 'w') as f:
    # print('Test samples number: ', len(bbox_iou))
    f.write('Test samples number: {}\n'.format(len(bbox_iou)))
    # print('Total wrong predictions for class: ', class_total_wrong_pred)
    f.write('Total wrong predictions for class: {}\n'.format(class_total_wrong_pred))
    # print('Total wrong predictions for attributes: ', attr_total_wrong_pred)
    f.write('Total wrong predictions for attributes: {}\n'.format(attr_total_wrong_pred))
    # print('bbox iou average: ', np.mean(bbox_iou))
    f.write('bbox iou average: {}\n'.format(np.mean(bbox_iou)))
    # print('Class prediction accuracy: ', np.mean(cls_acc)*100)
    f.write('Class prediction accuracy: {}\n'.format(np.mean(cls_acc)*100))
    # print('Attribute prediction accuracy: ', np.mean(attr_acc)*100)
    f.write('Attribute prediction accuracy: {}\n'.format(np.mean(attr_acc)*100))
    for i, cls in enumerate(class35):
        print('{}% Total of class {}: {}, missed: {}, wrong predictions: {}'.format('%.2f'%(cls_acc[i]*100), cls, class_total[i], class_er[i], (class_wrong_pred[i]*100)/class_total_wrong_pred))
        f.write('{}% Total of class {}: {}, missed: {}, wrong predictions: {}\n'.format('%.2f'%(cls_acc[i]*100), cls, class_total[i], class_er[i], (class_wrong_pred[i]*100)/class_total_wrong_pred))
    print('--------------------------------------------------------')
    f.write('--------------------------------------------------------\n')
    for i, attr in enumerate(attr200):
        print('{}% Total of attribute {}: {}, missed: {}, wrong predictions: {}'.format('%.2f'%(attr_acc[i]*100), attr_names[attr], attr_total[i], attr_er[i],(attr_wrong_pred[i]*100)/attr_total_wrong_pred))
        f.write('{}% Total of attribute {}: {}, missed: {}, wrong predictions: {}\n'.format('%.2f'%(attr_acc[i]*100), attr_names[attr], attr_total[i], attr_er[i],(attr_wrong_pred[i]*100)/attr_total_wrong_pred))
    print('Test samples number: ', len(bbox_iou))
    # f.write('Test samples number: {}\n'.format(len(bbox_iou)))
    print('Total wrong predictions for class: ', class_total_wrong_pred)
    # f.write('Total wrong predictions for class: {}\n'.format(class_total_wrong_pred))
    print('Total wrong predictions for attributes: ', attr_total_wrong_pred)
    # f.write('Total wrong predictions for attributes: {}\n'.format(attr_total_wrong_pred))
    print('bbox iou average: ', np.mean(bbox_iou))
    # f.write('bbox iou average: {}\n'.format(np.mean(bbox_iou)))
    print('Class prediction accuracy: ', np.mean(cls_acc)*100)
    # f.write('Class prediction accuracy: {}\n'.format(np.mean(cls_acc)*100))
    print('Attribute prediction accuracy: ', np.mean(attr_acc)*100)
    # f.write('Attribute prediction accuracy: {}\n'.format(np.mean(attr_acc)*100))   
    ret = model.evaluate(X['btl'], [X['bbwh'][:,:4], X['attr'], X['cls']])
    print('metrics_names: ', model.metrics_names,'evaluate: ', ret)
    
