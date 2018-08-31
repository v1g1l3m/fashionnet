import time
import os
import numpy as np
import glob
from random import randint, shuffle
import shutil
from PIL import Image
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from utils import init_globals
import logging
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")

### GLOBALS
batch_size = 64
img_width = 224             # For VGG16
img_height = 224            # For VGG16
img_channel = 3

fashion_dataset_path='../Data/fashion_data/'

dataset_path='../Data/dataset_iou'
dataset_train_path=os.path.join(dataset_path, 'train')
dataset_val_path=os.path.join(dataset_path, 'validation')
dataset_test_path=os.path.join(dataset_path, 'test')

# btl_path = 'bottleneck'
btl_path = 'E:\\ML\\bottleneck_iou'
btl_train_path = os.path.join(btl_path, 'train')
btl_val_path = os.path.join(btl_path, 'validation')
btl_test_path = os.path.join(btl_path, 'test')

# Create directory structure
def create_bottleneck_structure():
    if not os.path.exists(btl_path):
        os.makedirs(btl_path)
    if not os.path.exists(btl_train_path):
        os.makedirs(btl_train_path)
    if not os.path.exists(btl_val_path):
        os.makedirs(btl_val_path)
    if not os.path.exists(btl_test_path):
        os.makedirs(btl_test_path)

def save_bottleneck_iou(num_per_file):
    ## Build the VGG16 network
    model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    for train_val in ['validation', 'train']:
        img_type_iou_tuples = []
        with open(os.path.join(btl_path, 'btl_'+train_val+'.txt'), 'w') as f_image:
            for i, cloth_type in enumerate(type_names):
                dataset_train_class_path = os.path.join(dataset_path, train_val, cloth_type)
                logging.debug('dataset_train_class_path {}'.format(dataset_train_class_path))
                images_path_name = sorted(glob.glob(dataset_train_class_path + '/*.jpg'))
                for name in images_path_name:
                    if os.name == 'nt':
                        name = name.replace('\\', '/')
                    iou = np.float(name.split('_')[-1].split('.jpg')[0])
                    type_1_hot = np.zeros((len(type_names),))
                    type_1_hot[i] = iou
                    img_type_iou_tuples.append((name, type_1_hot))
                    f_image.write(str(name)+' '+cloth_type+' '+str(iou)+'\n')

        shuffle(img_type_iou_tuples)
        images_list = []
        type_1_hot_list = []
        index = 0
        with open(os.path.join(btl_path, 'btl_' + train_val + '_npz.txt'), 'w') as f_image:
            for name, type_1_hot in img_type_iou_tuples:
                current_size = len(images_list)
                img = Image.open(name)
                img = img.resize((img_width, img_height))
                img = np.array(img).astype(np.float32)
                images_list.append(img)
                type_1_hot_list.append(type_1_hot)
                if (current_size < num_per_file-1):
                    continue
                images_list = preprocess_input(np.array(images_list))
                type_1_hot_list = np.array(type_1_hot_list)
                bottleneck_features_train_class = model.predict(images_list, batch_size)
                btl_save_file_name = os.path.join(btl_path, train_val) + '/btl_' + train_val + '_' + \
                                    str(num_per_file) + '_' + str(index*num_per_file).zfill(7) + '.npz'
                logging.info('btl_save_file_name {}'.format(btl_save_file_name))
                np.savez(open(btl_save_file_name, 'wb'), btl=bottleneck_features_train_class,
                         iou3=type_1_hot_list)
                f_image.write(str(btl_save_file_name) + '\n')
                images_list = []
                type_1_hot_list = []
                index += 1

def save_bottleneck_cls_attr(num_per_file):

    for train_val in ['validation', 'train']:
        img_name_class_attr_tuples = []
        with open(os.path.join(btl_path, 'btl_' + train_val + '.txt'), 'w') as f_image:
            for class_name in class_names:
                dataset_train_class_path = os.path.join(dataset_path, train_val, class_name)
                logging.debug('dataset_train_class_path {}'.format(dataset_train_class_path))
                images_path_name = sorted(glob.glob(dataset_train_class_path + '/*.jpg'))
                for name in images_path_name:
                    if os.name == 'nt':
                        name = name.replace('\\', '/')
                    indx_str = name.split('_')[-1].split('.jpg')[0]
                    attrs_1_hot = np.zeros(1000,)
                    if len(indx_str) > 0:
                        attrs_indx = list(map(int, indx_str.split('-')))
                        attrs_1_hot[attrs_indx] = 1
                    class_1_hot = np.zeros((len(class_names),), dtype=np.float32)
                    class_1_hot[class_names.index(name.split('/')[-2])] = 1
                    img_name_class_attr_tuples.append((name, class_1_hot, attrs_1_hot))
                    f_image.write(str(name) + ' ' + str(name.split('/')[-2]) + ' ' + indx_str + '\n')

        ## Build the VGG16 network
        model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        shuffle(img_name_class_attr_tuples)
        images_list = []
        class_1_hot_list = []
        attrs_1_hot_list = []
        index = 0
        with open(os.path.join(btl_path, 'btl_' + train_val + '_npz.txt'), 'w') as f_image:
            for name, class_1_hot, attrs_1_hot in img_name_class_attr_tuples:
                current_size = len(images_list)
                img = Image.open(name)
                img = img.resize((img_width, img_height))
                img = np.array(img).astype(np.float32)
                images_list.append(img)
                class_1_hot_list.append(class_1_hot)
                attrs_1_hot_list.append(attrs_1_hot)
                if (current_size < num_per_file-1):
                    continue
                images_list = preprocess_input(np.array(images_list))
                class_1_hot_list = np.array(class_1_hot_list)
                attrs_1_hot_list = np.array(attrs_1_hot_list)
                bottleneck_features_train_class = model.predict(images_list, batch_size)
                btl_save_file_name = os.path.join(btl_path, train_val) + '/btl_' + train_val + '_' + \
                                    str(num_per_file) + '_' + str(index*num_per_file).zfill(7) + '.npz'
                logging.info('btl_save_file_name {}'.format(btl_save_file_name))
                np.savez(open(btl_save_file_name, 'wb'), btl=bottleneck_features_train_class,
                         cls=class_1_hot_list, attr=attrs_1_hot_list)
                f_image.write(str(btl_save_file_name) + '\n')
                images_list = []
                class_1_hot_list = []
                attrs_1_hot_list = []
                index += 1

if __name__ == '__main__':
    global class_names, input_shape, attr_names, type_names
    class_names, input_shape, attr_names = init_globals()
    type_names = ['upper-body', 'lower-body', 'full-body']
    create_bottleneck_structure()
    save_bottleneck_iou(64)