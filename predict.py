# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import shutil
from PIL import Image
import skimage
from colorthief import ColorThief
from keras.applications import VGG16
from keras.models import load_model
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from utils import init_globals, get_image_paths, draw_rect
from segmentation import cluster_bboxes, selective_search_bbox_fast
import logging
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")
import time
# import grpc
# from tensorflow.contrib.util import make_tensor_proto, make_ndarray
# from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input

### GLOBALS
batch_size = 64
img_width = 224             # For VGG16
img_height = 224            # For VGG16
img_channel = 3
prediction_path = '../prediction/'
results_path = os.path.join(prediction_path, 'results')
fashion_dataset_path='../Data/fashion_data/'

def predict(X):
    channel = grpc.insecure_channel('localhost:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'fashionnet'
    request.model_spec.signature_name = 'predict'
    request.inputs['images'].CopyFrom(make_tensor_proto(X, shape=X.shape))
    result = stub.Predict(request, 100.0)
    bboxes = make_ndarray(result.outputs['bbox'])
    attrs = make_ndarray(result.outputs['attr'])
    classes = make_ndarray(result.outputs['cls'])
    return (bboxes, attrs, classes)

def intersection_area(boxes1, boxes2):
    x11, y11, x12, y12 = boxes1[0], boxes1[1], boxes1[2], boxes1[3]
    x21, y21, x22, y22 = boxes2[0], boxes2[1], boxes2[2], boxes2[3]
    xA = np.maximum(x11, x21)
    yA = np.maximum(y11, y21)
    xB = np.minimum(x12, x22)
    yB = np.minimum(y12, y22)
    return np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

def get_palette(name, num_colors=5):
    color_thief = ColorThief(name)
    cf_palette = [(x[0], x[1], x[2]) for x in color_thief.get_palette(color_count=num_colors)]
    closest, _ = pairwise_distances_argmin_min(cf_palette, colors)
    cl_names = [color_names[i] for i in closest]
    return cl_names

def get_crops_resized(image_path_name, bboxeswh):
    img = Image.open(image_path_name)
    img_crops = []
    dims = []
    for index, bboxwh in enumerate(bboxeswh):
        x1, y1, x2, y2 = bboxwh[0], bboxwh[1], bboxwh[2]+bboxwh[0], bboxwh[3]+bboxwh[1]
        img_crop = img.crop((x1, y1, x2, y2))
        dims.append((x1, y1, img_crop.size[0], img_crop.size[1]))
        img_crop = img_crop.resize((img_width, img_height))
        img_crop = np.array(img_crop).astype(np.float32)
        if img_crop.shape[2] > 3:
            img_crop = img_crop[:,:,:3]
        img_crops.append(img_crop)
    return (img_crops, dims)

def display(image_path_name, width, height, bboxeswh, prediction_iou, prediction_class_name, prediction_class_prob, prediction_attr_names,
            prediction_attr_probs, prediction_bbox):
    thres = 50
    true_frames = []
    full_list_cand = []
    for i in range(len(prediction_bbox)):
        w1, h1 = prediction_bbox[i][2] - prediction_bbox[i][0], prediction_bbox[i][3] - prediction_bbox[i][1]
        if prediction_class_prob[i]*100 >= thres and prediction_class_name[i] != 'Фон':
            if w1 * h1 < (width * height) / 40:
                print('removed for size: ', prediction_bbox[i])
                continue
            full_list_cand.append((prediction_bbox[i], prediction_iou[i], prediction_class_name[i],
                                  prediction_class_prob[i], prediction_attr_names[i], prediction_attr_probs[i]))
            true_frames.append(bboxeswh[i])
    #----------------------------------CLUSTERISATION-----------------------------------------
    true_bboxes = np.array([x[0] for x in full_list_cand])
    true_bboxes_cls_probs = np.array([x[3] for x in full_list_cand])
    # np_true_bboxes_scaled = np.array([[bb[0] / width, bb[1] / height, bb[2] / width, bb[3] / height] for bb in true_bboxes])
    np_true_bboxes_scaled = np.array([[((bb[2]+bb[0])/2)/width, ((bb[3]+bb[1])/2)/height] for bb in true_bboxes])
    bbox_centers_colors = np.zeros((len(true_bboxes), 3))
    bbox_colors = np.zeros((len(true_bboxes), 3))
    bbox_colors[:, 0] = 1

    answer = []
    if len(true_bboxes) > 1:
        af = AffinityPropagation(preference=-0.05).fit(np_true_bboxes_scaled)
        labels = af.labels_
        for cluster in np.unique(labels):
            cluster_color = np.random.rand(3,)
            bbox_centers_colors[labels == cluster] = cluster_color

            cluster_cls_probs = true_bboxes_cls_probs[labels == cluster]
            index = np.argwhere(true_bboxes_cls_probs == np.max(cluster_cls_probs))[0][0]
            answer.append(full_list_cand[index])
    else:
        if len(true_bboxes) == 1:
            answer.append(full_list_cand[0])
    # --------------------INTERSECTIONS---------------------------------------
    candidates = []
    for i in range(len(answer)):
            bboxA = answer[i][0]
            x1, y1, w1, h1 = bboxA[0], bboxA[1], bboxA[2] - bboxA[0], bboxA[3] - bboxA[1]
            for j in range(len(answer)):
                bboxB = answer[j][0]
                x2, y2, w2, h2 = bboxB[0], bboxB[1], bboxB[2] - bboxB[0], bboxB[3] - bboxB[1]
                if j != i and intersection_area(bboxA, bboxB) > 0.5*w1*h1 and w2*h2 > w1*h1:
                    print('removed for intersect: ', bboxA)
                    break
            else:
                candidates.append(answer[i])

    fig, axes = plt.subplots(1, 3, figsize=(8, 5), frameon=False)
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    xlabel_ax1 = []
    img1 = Image.open(image_path_name)
    img2 = Image.open(image_path_name)
    img3 = Image.open(image_path_name)
    # 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111
    for i, bbox in enumerate(bboxeswh):
        clr = list(np.random.rand(3,))
        clr.append(0.8)
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        draw_rect(ax1, img1, (x, y, w, h), edgecolor=clr)
    ax1.imshow(img1)
    ax1.set_xlabel(image_path_name)
    # 2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
    for i, t in enumerate(full_list_cand):
        bbox, bbox_prob, cls_name, cls_prob, attr_name, attr_prob = t[0], t[1], t[2], t[3], t[4], t[5]
        x, y, w, h = bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        draw_rect(ax2, img2, (x, y, w, h), '%s %.3s%%'%(cls_name, cls_prob*100), edgecolor=bbox_colors[i])
        ax2.plot(x + w/2, y + h/2, c=bbox_centers_colors[i], marker='o')
        # draw_rect(ax2, img2, (x, y, w, h), '%s %.3s%%'%(cls_name, cls_prob*100))
        # ax2.plot(x + w / 2, y + h / 2, 'ro')
    ax2.imshow(img2, aspect='equal')
    # 3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
    img00 = Image.open(image_path_name)
    # img0 = Image.open(image_path_name)
    # fig0 = plt.figure(figsize=(5, 5), frameon=False)
    # fig0.set_size_inches(5, 5)
    # ax0 = plt.Axes(fig0, [0., 0., 1., 1.])
    # ax0.set_axis_off()
    # fig0.add_axes(ax0)
    ttags = []
    with open(os.path.join(results_path, 'annotation.txt'), 'a') as f:
        for bbox, bbox_prob, cls_name, cls_prob, attr_name, attr_prob in candidates:
            x, y, w, h = bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            draw_rect(ax3, img3, (x, y, w, h), cls_name, textcolor=(0, 255, 0))
            ax3.plot(x + w/2, y + h/2, 'ro')
            # draw_rect(ax0, img0, (x, y, w, h), cls_name, textcolor=(0, 255, 0))
            attr_probs = sorted(attr_prob, reverse=True)
            tags=','.join([attr_name[np.argwhere(attr_prob == x)[0][0]] for x in attr_probs])
            ttags.append(tags)
            # palette=','.join(get_palette(img00.crop((bbox[0],bbox[1],bbox[2],bbox[3]))))
            # f.write('{} {} {}\n'.format(os.path.split(image_path_name)[1], tags, palette))
    ax3.imshow(img3, aspect='equal')
    ax3.set_xlabel('\n'.join(ttags))
    plt.show()
    # fig0.savefig(os.path.join(results_path, os.path.split(image_path_name)[1]))

### MAIN ###
if __name__ == '__main__':
    global class_names, input_shape, attr_names, attr_names_RU, class_names_RU, class36, attr200, colors, color_names
    class_names, input_shape, attr_names = init_globals()
    class36 = ['None', 'Blazer', 'Top', 'Dress', 'Chinos', 'Jersey', 'Cutoffs', 'Kimono', 'Cardigan', 'Jeggings', 'Button-Down',
               'Romper', 'Skirt', 'Joggers', 'Tee', 'Turtleneck', 'Culottes', 'Coat', 'Henley', 'Jeans', 'Hoodie',
               'Blouse',
               'Tank', 'Shorts', 'Bomber', 'Jacket', 'Parka', 'Sweatpants', 'Leggings', 'Flannel', 'Sweatshorts',
               'Jumpsuit', 'Poncho', 'Trunks', 'Sweater', 'Robe']
    attr200 = [730, 365, 513, 495, 836, 596, 822, 254, 884, 142, 212, 883, 837, 892, 380, 353, 196, 546, 335, 162, 441,
               717,
               760, 568, 310, 705, 745, 81, 226, 830, 620, 577, 1, 640, 956, 181, 831, 720, 601, 112, 820, 935, 969,
               358,
               933, 983, 616, 292, 878, 818, 337, 121, 236, 470, 781, 282, 913, 93, 227, 698, 268, 61, 681, 713, 239,
               839,
               722, 204, 457, 823, 695, 993, 0, 881, 817, 571, 565, 770, 751, 692, 593, 825, 574, 50, 207, 186, 237,
               563,
               300, 453, 897, 944, 438, 688, 413, 409, 984, 191, 697, 368, 133, 676, 11, 754, 800, 83, 14, 786, 141,
               841,
               415, 608, 276, 998, 99, 851, 429, 287, 815, 437, 747, 44, 988, 249, 543, 560, 653, 843, 208, 899, 321,
               115,
               887, 699, 15, 764, 48, 749, 852, 811, 862, 392, 937, 87, 986, 129, 336, 689, 245, 911, 309, 775, 638,
               184,
               797, 512, 45, 682, 139, 306, 880, 231, 802, 264, 648, 410, 30, 356, 531, 982, 116, 599, 774, 900, 218,
               70,
               562, 108, 25, 450, 785, 877, 18, 42, 624, 716, 36, 920, 423, 784, 788, 538, 325, 958, 480, 20, 38, 931,
               666,
               561]
    class_names_RU = ['Фон']
    with open(os.path.join(fashion_dataset_path, 'Anno/1_list_category_cloth.txt'), encoding='utf-8') as f:
        next(f)
        next(f)
        for line in f:
            class_names_RU.append(line.split()[2])
    attr_names_RU = []
    with open(os.path.join(fashion_dataset_path, 'Anno/2_list_attr_cloth.txt'), encoding='utf-8') as f:
        next(f)
        next(f)
        for line in f:
            lines = line.split()
            for i in range(len(lines)):
                if lines[i].isdigit():
                    break
            attr_names_RU.append('-'.join(lines[i+1:]))
    color_names = []
    colors = []
    with open('../Data/color_table.txt') as f:
        for line in f:
            line = line.split()
            r, g, b = line[1][:2], line[1][2:4], line[1][4:]
            color_names.append(line[0]+'-#'+line[1])
            colors.append([int(r, 16), int(g, 16), int(b, 16)])
    colors = np.array(colors)

    # if os.path.exists(results_path):
        # shutil.rmtree(results_path) # quationable
    # os.makedirs(results_path)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = load_model('models/model2.h5')
    for index, img_path in enumerate(get_image_paths(prediction_path)):
        image = Image.open(img_path)
        w, h = image.size[0], image.size[1]
        res_value = 400
        image = image.resize((res_value, res_value))
        img_as_np_arr = np.array(image)
        if img_as_np_arr.shape[2] > 3:
            img_as_np_arr = image[:,:,:3]
        # bboxeswh = cluster_bboxes(selective_search_bbox_fast(image, w*h/50), w, h, -0.15)
        t1 = time.time()
        bb = selective_search_bbox_fast(img_as_np_arr, (w * h) / 40)
        t2 = time.time()
        logging.info('segmentation time: {}, number: {}'.format(t2 - t1, len(bb)))
        bboxeswh = cluster_bboxes(bb, w, h, res_value, res_value, preference=-0.35)
        image_crops, dims = get_crops_resized(img_path, bboxeswh)
        img = Image.open(img_path)
        img = img.resize((img_width, img_height))
        img = np.array(img).astype(np.float32)
        if img.shape[2] > 3:
            img = img[:,:,:3]
        image_crops.append(img)
        dims.append((0, 0, w, h))
        bboxeswh.append([0, 0, w, h])
        images_list = preprocess_input(np.array(image_crops))
        t1 = time.time()
        logging.info('prepare time: {}'.format(t1 - t2))
        bboxes, attrs, classes = model.predict(base_model.predict(images_list, verbose=1), verbose=1)
        t2 = time.time()
        logging.info('prediction time: {}'.format(t2 - t1))
        prediction_iou = []
        prediction_bbox = []
        prediction_attr_probs = []
        prediction_attr_names = []
        prediction_class_prob = []
        prediction_class_name = []
        for i, t in enumerate(zip(bboxes, classes, attrs)):
            pred_bbox, pred_cls, pred_attr = t[0], t[1], t[2]
            prediction_iou.append(0)
            prediction_bbox.append((dims[i][0] + pred_bbox[0]*dims[i][2], dims[i][1] + pred_bbox[1]*dims[i][3], dims[i][0] + pred_bbox[2]*dims[i][2],dims[i][1] +  pred_bbox[3]*dims[i][3]))
            if pred_cls[0] < 0.5:
                prediction_class_prob.append(np.max(pred_cls[1:]))
                prediction_class_name.append(class_names_RU[class_names.index(class36[np.argmax(pred_cls[1:])+1])])
            else:
                prediction_class_prob.append(pred_cls[0])
                prediction_class_name.append(class_names_RU[0])
            prediction_attr_probs.append([x for x in pred_attr if x>= 0.5])
            prediction_attr_names.append([attr_names[attr200[i]] for i in range(len(pred_attr)) if pred_attr[i] >= 0.5])
        display(img_path, w, h, bboxeswh, prediction_iou, prediction_class_name, prediction_class_prob, prediction_attr_names,
                prediction_attr_probs, prediction_bbox)
    a=2