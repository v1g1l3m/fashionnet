# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import shutil
from PIL import Image
import skimage
from keras.optimizers import *
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, model_from_json, load_model
from keras.layers import *
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from utils import init_globals, get_image_paths, draw_rect
from segmentation import selective_search_aggregated, cluster_bboxes
import logging
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")

### GLOBALS
batch_size = 64
img_width = 224             # For VGG16
img_height = 224            # For VGG16
img_channel = 3
top_model_path_load = 'output/bottleneck_fc_model'
prediction_path = '../prediction/'
results_path = os.path.join(prediction_path, 'results')
fashion_dataset_path='../Data/fashion_data/'

def model_iou():
    model_inputs = Input(shape=(7, 7, 512))
    common_inputs = model_inputs
    x = Flatten()(common_inputs)
    x = Dense(256, activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='tanh')(x)
    x = Dropout(0.5)(x)
    predictions_iou = Dense(1, activation='sigmoid', name='predictions_iou')(x)
    model = Model(inputs=model_inputs, outputs=predictions_iou)
    with open('best/iou0/bottleneck_fc_model.json', 'w') as f:
        f.write(model.to_json())
    model.load_weights('best/iou0/bottleneck_fc_model.h5', by_name=True)
    model.compile(optimizer=get_optimizer('Adadelta')[0], loss='mean_squared_error', metrics=['accuracy'])
    return model

def model_class():
    model_inputs = Input(shape=(7, 7, 512))
    common_inputs = model_inputs
    x = Flatten()(common_inputs)
    x = Dense(256, activation='tanh')(x)
    x = Dropout(0.5)(x)
    predictions_class = Dense(50, activation='softmax', name='predictions_class')(x)
    model = Model(inputs=model_inputs, outputs=predictions_class)
    model.load_weights('best/class0/bottleneck_fc_model.h5', by_name=True)
    model.compile(optimizer=get_optimizer('Adadelta')[0], loss='categorical_crossentropy', metrics=['accuracy'])
    with open('best/class0/bottleneck_fc_model.json', 'w') as f:
        f.write(model.to_json())
    return model

def create_one_model_from_two():

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # shape=(?, 7, 7, 512)
    model_inputs = base_model.input
    common_inputs = base_model.output

    ## Classes
    x = Flatten(name='f1')(common_inputs)
    x = Dense(256, activation='tanh', name='d1')(x)
    x = Dropout(0.5, name='dr1')(x)
    predictions_class = Dense(50, activation='softmax', name='predictions_class')(x)

    # iou
    x = Flatten(name='f2')(common_inputs)
    x = Dense(256, activation='tanh', name='d12')(x)
    x = Dropout(0.5, name='dr12')(x)
    x = Dense(256, activation='tanh', name='d22')(x)
    x = Dropout(0.5, name='dr22')(x)
    predictions_iou = Dense(1, activation='sigmoid', name='predictions_iou')(x)

    model = Model(inputs=model_inputs, outputs=[predictions_class, predictions_iou])
    print(model.summary())
    model.load_weights('bottleneck_fc_model.h5', by_name=True)
    model.compile(optimizer=get_optimizer('Adadelta')[0],
                  loss={'predictions_class': 'categorical_crossentropy', 'predictions_iou': 'mean_squared_error'},
                  metrics=['accuracy'])
    return model

def get_crops_resized(image_path_name, bboxeswh):
    img = Image.open(image_path_name)
    img_crops = []
    for index, bboxwh in enumerate(bboxeswh):
        x1, y1, x2, y2 = bboxwh[0], bboxwh[1], bboxwh[2]+bboxwh[0], bboxwh[3]+bboxwh[1]
        img_crop = img.crop((x1, y1, x2, y2))
        img_crop = img_crop.resize((img_width, img_height))
        img_crop = np.array(img_crop).astype(np.float32)
        img_crops.append(img_crop)
    return img_crops

def display(image_path_name, bboxes, prediction_iou, prediction_class_name, prediction_class_prob, prediction_attr_names,
                prediction_attr_probs):
    fig, axes = plt.subplots(1, 2, figsize=(8, 5), frameon=False)
    # ax1=axes[0]
    ax2=axes[0]
    ax3 = axes[1]
    img1 = Image.open(image_path_name)
    width, height = img1.size[0], img1.size[1]
    img2 = Image.open(image_path_name)
    img3 = Image.open(image_path_name)
    # for index, bbox in enumerate(bboxes):
    #     draw_rect(ax1, img1, bbox, '%.2s%%'%(prediction_iou[index]*100), edgecolor=np.random.rand(3,))
    # ax1.set_xlabel('Сегментация и результаты')

    true_bboxes = []
    probs = []
    cls_probs = []
    cls_names = []
    attr_nm = []
    attr_probs = []
    for i,b in enumerate(bboxes):
        if prediction_iou[i] >= 0.5:
            true_bboxes.append(b)
            probs.append(prediction_iou[i])
            cls_probs.append(prediction_class_prob[i])
            cls_names.append(prediction_class_name[i])
            attr_nm.append(prediction_attr_names[i])
            attr_probs.append(prediction_attr_probs[i])
    np_true_bboxes_scaled = np.array([[bb[0] / width, bb[1] / height, bb[2] / width, bb[3] / height] for bb in true_bboxes])
    bbox_centers_colors = np.zeros((len(np_true_bboxes_scaled), 3))
    bbox_colors = np.zeros((len(np_true_bboxes_scaled), 3))
    bbox_colors[:,0] = 1
    answer = []
    if len(true_bboxes) > 1:
        af = AffinityPropagation(preference=-0.5).fit(np_true_bboxes_scaled)
        labels = af.labels_
        bbox_sizes = [x[2] * x[3] for x in np_true_bboxes_scaled]
        for cluster in np.unique(labels):
            bboxes_cluster = np_true_bboxes_scaled[labels == cluster]
            cluster_color = np.random.rand(3,)
            bbox_sizes_cluster =[x[2]*x[3] for x in bboxes_cluster]
            min_size_bbox_index = np.argwhere(bbox_sizes == np.min(bbox_sizes_cluster))[0][0]
            bbox_centers_colors[labels == cluster] = cluster_color
            bbox_colors[min_size_bbox_index] = (0,1,1)
            km = KMeans(n_clusters=1).fit(bboxes_cluster)
            closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, bboxes_cluster)
            centroid = km.cluster_centers_[0]
            index = closest[0]
            answer.append(((centroid[0] * width, centroid[1] * height, centroid[2] * width, centroid[3] * height), cls_names[index], cls_probs[index], probs[index],attr_nm[index],attr_probs[index]))
    else:
        if len(true_bboxes) == 1:
            answer.append((true_bboxes[0], cls_names[0], cls_probs[0], probs[0],attr_nm[0],attr_probs[0]))

    for i, bbox in enumerate(true_bboxes):
        draw_rect(ax2, img2, bbox, '%s %.2s%% %.2s%%'%(cls_names[i],cls_probs[i]*100,probs[i]*100), edgecolor=bbox_colors[i])
        ax2.plot(bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2, c=bbox_centers_colors[i], marker='o')
    ax2.imshow(img2)

    # img0 = Image.open(image_path_name)
    # fig0 = plt.figure(figsize=(5, 5), frameon=False)
    # fig0.set_size_inches(5, 5)
    # ax0 = plt.Axes(fig0, [0., 0., 1., 1.])
    # ax0.set_axis_off()
    # fig0.add_axes(ax0)
    tags = []
    for bbox,name,prob,iou,attr,attr_prob in answer:
        attr_probs = sorted(attr_prob, reverse=True)[:5]
        tags.append(','.join([attr[np.argwhere(attr_prob == x)[0][0]] for x in attr_probs]))
        # draw_rect(ax0, img0, bbox, name, textcolor=(0, 255, 0))
        draw_rect(ax3, img3, bbox, name, textcolor=(0, 255, 0))
        ax3.plot(bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, 'ro')
    ax3.imshow(img3, aspect='equal')

    ax2.set_xlabel('Рамки, с вероятностью > 50%')
    ax3.set_xlabel('\n'.join(tags))
    plt.show()
    fig.savefig(os.path.join(results_path, 'sample-out-'+('%.7s'%random.random())[3:]+'.png'))

### MAIN ###
if __name__ == '__main__':
    global class_names, input_shape, attr_names, attr_names_RU, class_names_RU
    class_names, input_shape, attr_names = init_globals()
    class_names_RU = []
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
            attr_names_RU.append(' '.join(lines[i+1:]))
    if os.path.exists(results_path):
        shutil.rmtree(results_path) # quationable
    os.makedirs(results_path)
    images_path_names = []
    images_class = []
    images_bbox = []
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # with open('best/attr_class0/bottleneck_fc_model.json') as f:
    #     m2 = model_from_json(f.read())
    # m2.load_weights('best/attr_class0/bottleneck_fc_model.h5', by_name=True)
    # m2.compile(optimizer=get_optimizer('SGD', 0.001)[0],
    #            loss={'predictions_class': 'categorical_crossentropy', 'predictions_attr': 'binary_crossentropy'},
    #               metrics=['accuracy'])
    m2 = load_model('best/attr_class1/best-weights.hdf5')
    # with open('best/iou0/bottleneck_fc_model.json') as f:
    #     m1 = model_from_json(f.read())
    # m1.load_weights('best/iou0/bottleneck_fc_model.h5', by_name=True)
    # m1.compile(optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0), loss='mean_squared_error', metrics=['accuracy'])
    m1 = load_model('best/iou1/best-weights.hdf5')
    for index, img_path in enumerate(get_image_paths(prediction_path)):
        w, h, ch = skimage.io.imread(img_path).shape
        bboxeswh = cluster_bboxes(selective_search_aggregated(img_path), w, h, -0.05)
        image_crops = get_crops_resized(img_path, bboxeswh)
        images_list = preprocess_input(np.array(image_crops))
        predictions = base_model.predict(images_list, batch_size)
        predictions_iou = m1.predict(predictions, batch_size, verbose=1)
        predictions_cls, predictions_attr = m2.predict(predictions, batch_size, verbose=1)
        prediction_iou = []
        prediction_attr_probs = []
        prediction_attr_names = []
        prediction_class_prob = []
        prediction_class_name = []
        for pred_iou, pred_cls, pred_attr in zip(predictions_iou, predictions_cls, predictions_attr):
            prediction_iou.append(np.max(pred_iou))
            prediction_class_prob.append(np.max(pred_cls))
            prediction_class_name.append(class_names_RU[np.argmax(pred_cls)])
            prediction_attr_probs.append(pred_attr)
            prediction_attr_names.append([attr_names_RU[i] for i in range(len(pred_attr))])
        display(img_path, bboxeswh, prediction_iou, prediction_class_name, prediction_class_prob, prediction_attr_names,
                prediction_attr_probs)
    a=2