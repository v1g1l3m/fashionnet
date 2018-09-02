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
from dataset_create import get_gt_bbox_from_file, get_second_arg_from_file
import logging
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")

### GLOBALS
batch_size = 64
img_width = 224             # For VGG16
img_height = 224            # For VGG16
img_channel = 3

fashion_dataset_path='../Data/fashion_data/'

dataset_path='../Data/dataset_3heads100'
dataset_train_path=os.path.join(dataset_path, 'train')
dataset_val_path=os.path.join(dataset_path, 'validation')
dataset_test_path=os.path.join(dataset_path, 'test')

# btl_path = 'bottleneck'
btl_path = 'E:\\ML\\bottleneck_test'
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

def save_bottleneck_df(num_per_file):
    count = -1
    img_name_class_attr_bbox_part_tuples = []
    with open(fashion_dataset_path + '/Anno/list_bbox.txt') as file_bbox:
        with open(fashion_dataset_path + '/Anno/list_category_img.txt') as file_category:
            with open(os.path.join(fashion_dataset_path, 'Eval/list_eval_partition.txt')) as file_partition:
                    with open(os.path.join(fashion_dataset_path, 'Anno/list_attr_img.txt')) as file_attr:
                        next(file_attr)
                        next(file_attr)
                        for line in file_attr:
                            count += 1
                            line = line.split()
                            img_path = line[0]
                            img_attr = np.array(eval('[' + ','.join(line[1:]) + ']'))
                            img_attr = [x[0] for x in np.argwhere(img_attr == 1)]
                            img_part = get_second_arg_from_file(img_path, file_partition)
                            if img_part == 'val':
                                img_part = 'validation'
                            img_class_indx = get_second_arg_from_file(img_path, file_category)
                            img_class = class_names[int(img_class_indx) - 1]
                            img_gt_bbox = get_gt_bbox_from_file(img_path, file_bbox)
                            img_name_class_attr_bbox_part_tuples.append((img_path, img_class, img_attr, img_gt_bbox, img_part))
                            
    shuffle(img_name_class_attr_bbox_part_tuples)
     ## Build the VGG16 network
    model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)                                            
    for train_val in ['train', 'validation','test']:
        images_list = []
        class_1_hot_list = []
        attrs_1_hot_list = []
        bbox_list = []
        index = 0
        with open(os.path.join(btl_path, 'btl_' + train_val + '_npz.txt'), 'w') as f_image:
            for name, cls, attrs, bbox, part in img_name_class_attr_bbox_part_tuples:
                if part != train_val:
                    continue
                current_size = len(images_list)
                img = Image.open(os.path.join(fashion_dataset_path, 'Img', name))
                width, height = img.size[0], img.size[1]
                img = img.resize((img_width, img_height))
                img = np.array(img).astype(np.float32)
                images_list.append(img)
                class_1_hot = np.zeros((len(class35),), dtype=np.float32)
                if cls in class35:
                    class_1_hot[class35.index(cls)] = 1
                class_1_hot_list.append(class_1_hot)
                attrs_1_hot = np.zeros(200,)
                if len(attrs) > 0:
                    for x in attrs:
                        if x in attr200:
                            attrs_1_hot[attr200.index(x)] = 1
                attrs_1_hot_list.append(attrs_1_hot)
                bbox_list.append((bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height))
                if (current_size < num_per_file-1):
                    continue
                images_list = preprocess_input(np.array(images_list))
                class_1_hot_list = np.array(class_1_hot_list)
                attrs_1_hot_list = np.array(attrs_1_hot_list)
                bbox_list = np.array(bbox_list)
                bottleneck_features_train_class = model.predict(images_list, batch_size)
                btl_save_file_name = os.path.join(btl_path, train_val) + '/btl_' + train_val + '_' + \
                                    str(num_per_file) + '_' + str(index*num_per_file).zfill(7) + '.npz'
                logging.info('btl_save_file_name {}'.format(btl_save_file_name))
                np.savez(open(btl_save_file_name, 'wb'), btl=bottleneck_features_train_class,
                         cls=class_1_hot_list, attr=attrs_1_hot_list, bb=bbox_list)
                f_image.write(str(btl_save_file_name) + '\n')
                images_list = []
                class_1_hot_list = []
                attrs_1_hot_list = []
                bbox_list = []
                index += 1
                            
     
                
def save_bottleneck_3heads(num_per_file):

    for train_val in ['validation', 'train']:
        img_name_class_attr_iou_tuples = []
        with open(os.path.join(btl_path, 'btl_' + train_val + '.txt'), 'w') as f_image:
            for class_name in class_names:
                dataset_train_class_path = os.path.join(dataset_path, train_val, class_name)
                logging.debug('dataset_train_class_path {}'.format(dataset_train_class_path))
                images_path_name = sorted(glob.glob(dataset_train_class_path + '/*.jpg'))
                for name in images_path_name:
                    if os.name == 'nt':
                        name = name.replace('\\', '/')
                    iou = np.float(name.split('_')[-1].split('.jpg')[0])
                    indx_str = name.split('_')[-2].split('.jpg')[0]
                    attrs_1_hot = np.zeros(200,)
                    if len(indx_str) > 0:
                        attrs_indx = list(map(int, indx_str.split('-')))
                        for x in attrs_indx:
                            if x in attr200:
                                attrs_1_hot[attr200.index(x)] = 1
                    class_1_hot = np.zeros((len(class35),), dtype=np.float32)
                    img_class = name.split('/')[-2]
                    if img_class in class35:
                        class_1_hot[class35.index(img_class)] = 1
                    img_name_class_attr_iou_tuples.append((name, class_1_hot, attrs_1_hot, iou))
                    f_image.write(str(name) + ' ' + str(name.split('/')[-2]) + ' ' + indx_str + ' ' + str(iou) + '\n')

        ## Build the VGG16 network
        model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        shuffle(img_name_class_attr_iou_tuples)
        images_list = []
        class_1_hot_list = []
        attrs_1_hot_list = []
        iou_list = []
        index = 0
        t1=time.time()
        with open(os.path.join(btl_path, 'btl_' + train_val + '_npz.txt'), 'w') as f_image:
            for name, class_1_hot, attrs_1_hot, iou in img_name_class_attr_iou_tuples:
                current_size = len(images_list)
                img = Image.open(name)
                img = img.resize((img_width, img_height))
                img = np.array(img).astype(np.float32)
                images_list.append(img)
                class_1_hot_list.append(class_1_hot)
                attrs_1_hot_list.append(attrs_1_hot)
                iou_list.append(iou)
                if (current_size < num_per_file-1):
                    continue
                images_list = preprocess_input(np.array(images_list))
                class_1_hot_list = np.array(class_1_hot_list)
                attrs_1_hot_list = np.array(attrs_1_hot_list)
                iou_list = np.array(iou_list)
                t1 = time.time()
                bottleneck_features_train_class = model.predict(images_list,batch_size)
                print('total time: {}'.format(time.time()-t1))
                btl_save_file_name = os.path.join(btl_path, train_val) + '/btl_' + train_val + '_' + \
                                    str(num_per_file) + '_' + str(index*num_per_file).zfill(7) + '.npz'
                logging.info('btl_save_file_name {}'.format(btl_save_file_name))
                np.savez(open(btl_save_file_name, 'wb'), btl=bottleneck_features_train_class,
                         cls=class_1_hot_list, attr=attrs_1_hot_list, iou=iou_list)
                f_image.write(str(btl_save_file_name) + '\n')
                images_list = []
                class_1_hot_list = []
                attrs_1_hot_list = []
                iou_list = []
                index += 1
                # if index==6:
                    # break
        # print('time={}'.format(time.time()-t1))
        # break
                
if __name__ == '__main__':
    global class_names, input_shape, attr_names, type_names, attr200, class35
    class_names, input_shape, attr_names = init_globals()
    type_names = ['upper-body', 'lower-body', 'full-body']
    class35 = ['Blazer', 'Top', 'Dress', 'Chinos', 'Jersey', 'Cutoffs', 'Kimono', 'Cardigan', 'Jeggings', 'Button-Down', 'Romper', 'Skirt', 'Joggers', 'Tee', 'Turtleneck', 'Culottes', 'Coat', 'Henley', 'Jeans', 'Hoodie', 'Blouse', 'Tank', 'Shorts', 'Bomber', 'Jacket', 'Parka', 'Sweatpants', 'Leggings', 'Flannel', 'Sweatshorts', 'Jumpsuit', 'Poncho', 'Trunks', 'Sweater', 'Robe']
    attr200 = [730, 365, 513, 495, 836, 596, 822, 254, 884, 142, 212, 883, 837, 892, 380, 353, 196, 546, 335, 162, 441, 717, 760, 568, 310, 705, 745, 81, 226, 830, 620, 577, 1, 640, 956, 181, 831, 720, 601, 112, 820, 935, 969, 358, 933, 983, 616, 292, 878, 818, 337, 121, 236, 470, 781, 282, 913, 93, 227, 698, 268, 61, 681, 713, 239, 839, 722, 204, 457, 823, 695, 993, 0, 881, 817, 571, 565, 770, 751, 692, 593, 825, 574, 50, 207, 186, 237, 563, 300, 453, 897, 944, 438, 688, 413, 409, 984, 191, 697, 368, 133, 676, 11, 754, 800, 83, 14, 786, 141, 841, 415, 608, 276, 998, 99, 851, 429, 287, 815, 437, 747, 44, 988, 249, 543, 560, 653, 843, 208, 899, 321, 115, 887, 699, 15, 764, 48, 749, 852, 811, 862, 392, 937, 87, 986, 129, 336, 689, 245, 911, 309, 775, 638, 184, 797, 512, 45, 682, 139, 306, 880, 231, 802, 264, 648, 410, 30, 356, 531, 982, 116, 599, 774, 900, 218, 70, 562, 108, 25, 450, 785, 877, 18, 42, 624, 716, 36, 920, 423, 784, 788, 538, 325, 958, 480, 20, 38, 931, 666, 561]
    create_bottleneck_structure()
    save_bottleneck_3heads(1024)