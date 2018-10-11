import os
import pickle
import numpy as np
import glob
import fnmatch
from random import randint, shuffle, choice
import shutil
import logging

import skimage
from sklearn.utils import class_weight

logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")
from PIL import Image
from utils import init_globals, bb_intersection_over_union, get_attr300
from segmentation import selective_search_bbox_fast, cluster_bboxes
from multiprocessing import Pool
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import apply_affine_transform, flip_axis

fashion_dataset_path = '/media/star/3C4C65AA4C65601E/dev/deepfashion/fashion_data/'

def get_second_arg_from_file(image_path_name, file_ptr):
    for line in file_ptr:
        if image_path_name in line:
            dataset_split_name = line.split()[1]
            return dataset_split_name.strip()


def get_gt_bbox_from_file(image_path_name, file_ptr):
    for line in file_ptr:
        if image_path_name in line:
            x1 = int(line.split()[1])
            y1 = int(line.split()[2])
            x2 = int(line.split()[3])
            y2 = int(line.split()[4])
            bbox = [x1, y1, x2, y2]
            return bbox


def calculate_bbox_score_and_save_img(image_path_name, image_save_path, gt_bbox, img_type, attrs_str):
    def non_blocking_save(image, path):
        try:
            image.save(path)
        except:
            try:
                os.remove(path)
            except:
                pass

    img_read = Image.open(image_path_name)
    width, height = img_read.size[0], img_read.size[1]
    candidates = cluster_bboxes(selective_search_bbox_fast(np.array(img_read), int((width * height) / 50)), width,
                                height, preference=-0.35, fast=True)
    for x, y, w, h in (candidates):
        boxA = gt_bbox
        boxB = (x, y, x + w, y + h)
        iou = bb_intersection_over_union(boxA, boxB)
        img_crop = img_read.crop((x, y, x + w, y + h))
        image_save_name = image_path_name.split('/')[-2] + '_' + image_path_name.split('/')[-1].split('.')[
            0] + '_' + img_type + '_' + attrs_str + '_'
        image_save_path_name = image_save_path + '/' + image_save_name + str(iou) + '.jpg'
        logging.debug('image_save_path_name {}'.format(image_save_path_name))
        non_blocking_save(img_crop, image_save_path_name)
    # Ground Truth
    img_crop = img_read.crop(gt_bbox)
    image_save_name = image_path_name.split('/')[-2] + '_' + image_path_name.split('/')[-1].split('.')[
        0] + '_' + img_type + '_' + attrs_str + '_'
    image_save_path_name = image_save_path + '/' + image_save_name + '1.0' + '.jpg'
    logging.debug('image_save_path_name {}'.format(image_save_path_name))
    non_blocking_save(img_crop, image_save_path_name)


def generate_dataset_three_heads_from_indexes(indexes):
    count = -1
    with open(fashion_dataset_path + '/Anno/list_bbox.txt') as file_bbox:
        with open(fashion_dataset_path + '/Anno/list_category_img.txt') as file_category:
            with open(os.path.join(fashion_dataset_path, 'Eval/list_eval_partition.txt')) as file_partition:
                with open(os.path.join(fashion_dataset_path, 'Anno/list_attr_img.txt')) as file_attr:
                    next(file_attr)
                    next(file_attr)
                    for line in file_attr:
                        count += 1
                        if count not in indexes:
                            continue
                        line = line.split()
                        img_path = line[0]
                        img_attr = np.array(eval('[' + ','.join(line[1:]) + ']'))
                        attrs_str = '-'.join(map(str, [x[0] for x in list(np.argwhere(img_attr == 1))]))
                        img_part = get_second_arg_from_file(img_path, file_partition)
                        if img_part == 'val':
                            img_part = 'validation'
                        img_class_indx = get_second_arg_from_file(img_path, file_category)
                        img_class = class_names[int(img_class_indx) - 1]
                        img_gt_bbox = get_gt_bbox_from_file(img_path, file_bbox)
                        img_type_indx = str(class_cloth_type[img_class])
                        image_save_path = os.path.join(dataset_path, img_part, img_class)
                        calculate_bbox_score_and_save_img(os.path.join(fashion_dataset_path, 'Img', img_path),
                                                          image_save_path, img_gt_bbox, img_type_indx, attrs_str)
                        logging.info('{} - {}'.format(count, img_path.split('/')[-2] + '_' +
                                                      img_path.split('/')[-1].split('.')[0] + '_' +
                                                      img_type_indx + '_' + attrs_str + '_' + '.jpg'))


def generate_dataset_three_heads(job):
    start, stop = job[0], job[1]
    count = -1
    with open(fashion_dataset_path + '/Anno/list_bbox.txt') as file_bbox:
        with open(fashion_dataset_path + '/Anno/list_category_img.txt') as file_category:
            with open(os.path.join(fashion_dataset_path, 'Eval/list_eval_partition.txt')) as file_partition:
                with open(os.path.join(fashion_dataset_path, 'Anno/list_attr_img.txt')) as file_attr:
                    next(file_attr)
                    next(file_attr)
                    for line in file_attr:
                        count += 1
                        if count < start or count > stop:
                            continue
                        line = line.split()
                        img_path = line[0]
                        img_part = get_second_arg_from_file(img_path, file_partition)
                        if img_part != 'train':  # only training part
                            continue
                        if img_part == 'val':
                            img_part = 'validation'
                        img_attr = np.array(eval('[' + ','.join(line[1:]) + ']'))
                        attrs_str = '-'.join(map(str, [x[0] for x in list(np.argwhere(img_attr == 1))]))
                        img_class_indx = get_second_arg_from_file(img_path, file_category)
                        img_class = class_names[int(img_class_indx) - 1]
                        img_gt_bbox = get_gt_bbox_from_file(img_path, file_bbox)
                        img_type_indx = str(class_cloth_type[img_class])
                        image_save_path = os.path.join(dataset_path, img_part, img_class)
                        calculate_bbox_score_and_save_img(os.path.join(fashion_dataset_path, 'Img', img_path),
                                                          image_save_path, img_gt_bbox, img_type_indx, attrs_str)
                        logging.info('{} - {} - {}'.format(count, img_path.split('/')[-2] + '_' +
                                                           img_path.split('/')[-1].split('.')[0] + '_' +
                                                           img_type_indx + '_' + attrs_str + '_' + '.jpg',
                                                           image_save_path))

if __name__ == '__main__':
    global class_names, input_shape, attr_names, class_cloth_type, type_names, class35, attr200, attr300, F
    class_names, input_shape, attr_names = init_globals(fashion_dataset_path)
    class35 = ['Blazer', 'Top', 'Dress', 'Chinos', 'Jersey', 'Cutoffs', 'Kimono', 'Cardigan', 'Jeggings', 'Button-Down',
               'Romper', 'Skirt', 'Joggers', 'Tee', 'Turtleneck', 'Culottes', 'Coat', 'Henley', 'Jeans', 'Hoodie',
               'Blouse', 'Tank', 'Shorts', 'Bomber', 'Jacket', 'Parka', 'Sweatpants', 'Leggings', 'Flannel',
               'Sweatshorts', 'Jumpsuit', 'Poncho', 'Trunks', 'Sweater', 'Robe']
    attr200 = [730, 365, 513, 495, 836, 596, 822, 254, 884, 142, 212, 883, 837, 892, 380, 353, 196, 546, 335, 162, 441,
               717, 760, 568, 310, 705, 745, 81, 226, 830, 620, 577, 1, 640, 956, 181, 831, 720, 601, 112, 820, 935,
               969, 358, 933, 983, 616, 292, 878, 818, 337, 121, 236, 470, 781, 282, 913, 93, 227, 698, 268, 61, 681,
               713, 239, 839, 722, 204, 457, 823, 695, 993, 0, 881, 817, 571, 565, 770, 751, 692, 593, 825, 574, 50,
               207, 186, 237, 563, 300, 453, 897, 944, 438, 688, 413, 409, 984, 191, 697, 368, 133, 676, 11, 754, 800,
               83, 14, 786, 141, 841, 415, 608, 276, 998, 99, 851, 429, 287, 815, 437, 747, 44, 988, 249, 543, 560, 653,
               843, 208, 899, 321, 115, 887, 699, 15, 764, 48, 749, 852, 811, 862, 392, 937, 87, 986, 129, 336, 689,
               245, 911, 309, 775, 638, 184, 797, 512, 45, 682, 139, 306, 880, 231, 802, 264, 648, 410, 30, 356, 531,
               982, 116, 599, 774, 900, 218, 70, 562, 108, 25, 450, 785, 877, 18, 42, 624, 716, 36, 920, 423, 784, 788,
               538, 325, 958, 480, 20, 38, 931, 666, 561]
    attr350 = [483, 757, 736, 469, 714, 609, 222, 291, 361, 893, 68, 746, 23, 642, 303, 397, 873, 658, 806, 540, 84, 940, 411, 354, 520, 821, 293, 210, 284, 930, 17, 43, 687, 628, 967, 150, 921, 987, 827, 977, 671, 812, 104, 393, 654, 39, 246, 476, 73, 132, 307, 119, 872, 203, 725, 277, 869, 72, 999, 273, 763, 756, 359, 389, 449, 842, 114, 532, 777, 974, 446, 799, 146, 416, 110, 474, 936, 19, 848, 224, 202, 489, 907, 669, 328, 396, 90, 283, 840, 279, 667, 183, 360, 544, 694, 696, 662, 917, 431, 302, 735, 941, 901, 708, 124, 188, 891, 468, 948, 414, 272, 619, 24, 889, 482, 929, 324, 201, 567, 971, 902, 768, 131, 327, 262, 701, 793, 569, 189, 871, 159, 154, 58, 947, 47, 943, 618, 723, 970, 683, 909, 960, 674, 27, 854, 691, 153, 330, 953, 649, 561, 666, 931, 38, 20, 480, 958, 325, 538, 788, 784, 423, 920, 36, 716, 624, 42, 18, 877, 785, 450, 25, 108, 562, 70, 218, 900, 774, 599, 116, 982, 531, 356, 30, 410, 648, 264, 802, 231, 880, 306, 139, 682, 45, 512, 797, 184, 638, 775, 309, 911, 245, 689, 336, 129, 986, 87, 937, 392, 862, 811, 852, 749, 48, 764, 15, 699, 887, 115, 321, 899, 208, 843, 653, 560, 543, 249, 988, 44, 747, 437, 815, 287, 429, 851, 99, 998, 276, 608, 415, 841, 141, 786, 14, 83, 800, 754, 11, 676, 133, 368, 697, 191, 984, 409, 413, 688, 438, 944, 897, 453, 300, 563, 237, 186, 207, 50, 574, 825, 593, 692, 751, 770, 565, 571, 817, 881, 0, 993, 695, 823, 457, 204, 722, 839, 239, 713, 681, 61, 268, 698, 227, 93, 913, 282, 781, 470, 236, 121, 337, 818, 878, 292, 616, 983, 933, 358, 969, 935, 820, 112, 601, 720, 831, 181, 956, 640, 1, 577, 620, 830, 226, 81, 745, 705, 310, 568, 760, 717, 441, 162, 335, 546, 196, 353, 380, 892, 837, 883, 212, 142, 884, 254, 822, 596, 836, 495, 513, 365, 730]
    attr300, F = get_attr300()

    # ---------------------3 HEADS WHOLE--------------------------
    # # with open(fashion_dataset_path + 'Anno/list_category_img.txt') as f:
    # #     total_count = int(f.readline())
    # # b = total_count // num_proc
    # # jobs = [((num_proc-1)*b, total_count)]
    # # for i in range(num_proc - 1):
    # #     jobs.append((i*b, (i+1)*b))
    # # print(jobs)
    # p = Pool(num_proc)
    # jobs = [(240100, 289222), (23334, 72305), (95870, 144610), (167744, 216915)]
    # p.map(generate_dataset_three_heads, jobs)

    # --------------------DeepFashion dataset stats---------------------
    # with open(fashion_dataset_path + 'deepfashion.txt', 'w') as f:
    #     img_path_bbox_attr_cls_tuples_list = []
    #     with open(fashion_dataset_path + '/Anno/list_bbox.txt') as file_bbox:
    #         with open(fashion_dataset_path + '/Anno/list_category_img.txt') as file_category:
    #             # with open(os.path.join(fashion_dataset_path, 'Eval/list_eval_partition.txt')) as file_partition:
    #             with open(os.path.join(fashion_dataset_path, 'Anno/list_attr_img.txt')) as file_attr:
    #                 next(file_attr)
    #                 next(file_attr)
    #                 for line in file_attr:
    #                     line = line.split()
    #                     img_path = os.path.join(fashion_dataset_path, 'Img', line[0])
    #                     img = Image.open(img_path)
    #                     width, height = img.size[0], img.size[1]
    #                     img_attr = np.array(eval('[' + ','.join(line[1:]) + ']'))
    #                     img_attr = '-'.join(map(str, [x[0] for x in list(np.argwhere(img_attr == 1))]))
    #                     if len(img_attr) == 0:
    #                         img_attr = None
    #                     img_class = class_names[int(get_second_arg_from_file(line[0], file_category)) - 1]
    #                     img_gt_bbox = get_gt_bbox_from_file(line[0], file_bbox)
    #                     img_gt_bbox = [img_gt_bbox[0] / width, img_gt_bbox[1] / height, img_gt_bbox[2] / width,
    #                                    img_gt_bbox[3] / height]
    #                     img_gt_bbox = '-'.join(map(str, img_gt_bbox))
    #                     # img_part = get_second_arg_from_file(line[0], file_partition)
    #                     # if img_part == 'val':
    #                     #     img_part = 'validation'
    #                     f.write('{} {} {} {} {} {}\n'.format(img_path, img_gt_bbox, img_attr, img_class, width, height))
    #                     img_path_bbox_attr_cls_tuples_list.append(
    #                         (img_path, img_gt_bbox, img_attr, img_class, width, height))
    #
        # for root, dirnames, filenames in sorted(os.walk('fashion_data\\neg_class')):
        #     for filename in sorted(fnmatch.filter(filenames, '*.*g')):
        #         full_path = os.path.join(root, filename)
        #         img = Image.open(full_path)
        #         width, height = img.size[0], img.size[1]
        #         f.write('{} {} {} {} {} {}\n'.format(full_path, 'None', 'None', 'None', width, height))
    # --------------------------------------------------------------------------------------------------------
    # img_path_bbox_attr_cls_tuples_list = []
    # with open(fashion_dataset_path + 'deepfashion.txt') as f:
    #     for line in f:
    #         line = line.split()
    #         if line[1] == 'None' or line[2] == 'None':
    #             continue
    #         img_path_bbox_attr_cls_tuples_list.append((line[0], line[1], line[2], line[3], line[4], line[5]))
    # # ---------------------------------------------------------------------------------------------------------
    # attrN = attr350
    # crop = 2420
    # attr_count = dict(((x, []) for x in range(len(attrN))))
    # for i in range(len(img_path_bbox_attr_cls_tuples_list)):
    #     line = img_path_bbox_attr_cls_tuples_list[i][2]
    #     for y in list(map(int, line.split('-'))):
    #         if y in attrN:
    #             attr_count[attrN.index(y)].append(i)
    # attr_count_list = sorted([(len(attr_count[i]), i) for i in range(len(attrN))])
    # img_gen = ImageDataGenerator(rotation_range=20,
    #                              horizontal_flip=True,
    #                              width_shift_range=0.1,
    #                              height_shift_range=0.1,
    #                              shear_range=0.05,
    #                              zoom_range=0.05)
    # attr_count_crop = dict(((x, 0) for x in range(len(attrN))))
    #
    # with open(fashion_dataset_path + 'augmented_'+str(len(attrN))+'_'+str(crop)+'.txt', 'w') as f:
    #     for value, key in attr_count_list:
    #         if attr_count_crop[key] >= crop:
    #             continue
    #         dir = str(key)
    #         if not os.path.exists(fashion_dataset_path + 'augmented/'+dir):
    #             os.mkdir(fashion_dataset_path + 'augmented/'+dir)
    #         if len(attr_count[key]) > crop:
    #             shuffle(attr_count[key])
    #             attr_count[key] = attr_count[key][:crop]
    #
    #         for i in attr_count[key]:
    #             save_path = fashion_dataset_path + 'augmented/'+dir+'/img'+str(attr_count_crop[key]).zfill(8)+'.jpg'
    #             shutil.copy(img_path_bbox_attr_cls_tuples_list[i][0].replace('\\', '/'), save_path)
    #             line = list(map(int, img_path_bbox_attr_cls_tuples_list[i][2].split('-')))
    #             attr_list = [attrN.index(x) for x in line if x in attrN]
    #             f.write('{} {} {} {} {} {}\n'.format(save_path, img_path_bbox_attr_cls_tuples_list[i][1], '-'.join(map(str, attr_list)), img_path_bbox_attr_cls_tuples_list[i][3], img_path_bbox_attr_cls_tuples_list[i][4], img_path_bbox_attr_cls_tuples_list[i][5]))
    #             for y in attr_list:
    #                 attr_count_crop[y] += 1
    #             if attr_count_crop[key] >= crop:
    #                 break
    #
    #         while attr_count_crop[key] < crop:
    #             ii = choice(attr_count[key])
    #             arr = np.array(skimage.io.imread(img_path_bbox_attr_cls_tuples_list[ii][0].replace('\\', '/')))
    #             w, h = int(img_path_bbox_attr_cls_tuples_list[ii][4]), int(img_path_bbox_attr_cls_tuples_list[ii][5])
    #             bbox = [float(x) for x in img_path_bbox_attr_cls_tuples_list[ii][1].split('-')]
    #             if len(arr.shape) != 3 or arr.shape[2] != 3:
    #                 continue
    #             transform_parameters = img_gen.get_random_transform(arr.shape, seed=attr_count_crop[key])
    #             res = img_gen.random_transform(arr, seed=attr_count_crop[key])
    #             x = np.zeros((h, w, 3))
    #             x[int(bbox[1]*h):int(bbox[3]*h)-1, int(bbox[0]*w):int(bbox[2]*w)-1, 0] = 100
    #             x = apply_affine_transform(x, transform_parameters.get('theta', 0),
    #                                        transform_parameters.get('tx', 0),
    #                                        transform_parameters.get('ty', 0),
    #                                        transform_parameters.get('shear', 0),
    #                                        transform_parameters.get('zx', 1),
    #                                        transform_parameters.get('zy', 1),
    #                                        row_axis=0, col_axis=1,
    #                                        channel_axis=2,
    #                                        fill_mode='nearest', cval=0.)
    #             if transform_parameters.get('flip_horizontal', False):
    #                 x = flip_axis(x, 1)
    #             if transform_parameters.get('flip_vertical', False):
    #                 x = flip_axis(x, 0)
    #             x = x[:, :, 0]
    #             arr_h = np.max(x, axis=1); arr_w = np.max(x, axis=0)
    #             i=0;
    #             while(i <= h  and arr_h[i]<1e-14):
    #                 i+=1
    #             y1=i;i=h-1;
    #             while(i >= 0 and arr_h[i]<1e-14):
    #                 i-=1
    #             y2=i;i=0;
    #             while(i <= w  and arr_w[i]<1e-14):
    #                 i+=1
    #             x1=i;i=w-1;
    #             while(i >= 0 and arr_w[i]<1e-14):
    #                 i-=1
    #             x2=i;
    #             bbox = (x1/w, y1/h, x2/w, y2/h)
    #             line = list(map(int, img_path_bbox_attr_cls_tuples_list[ii][2].split('-')))
    #             attr_list = [attrN.index(x) for x in line if x in attrN]
    #             cls = img_path_bbox_attr_cls_tuples_list[ii][3]
    #             path = fashion_dataset_path + 'augmented/'+dir+'/img'+str(attr_count_crop[key]).zfill(8)+'.jpg'
    #             for y in attr_list:
    #                 attr_count_crop[y] += 1
    #             im = Image.fromarray(res)
    #             im.save(path)
    #             f.write('{} {} {} {} {} {}\n'.format(path, '-'.join(map(str, bbox)), '-'.join(map(str, attr_list)), cls, w, h))
    #
    # aaa = sorted([(attr_count_crop[i], i) for i in range(len(attrN))])
    # for val, key in aaa:
    #     print(key, val)
    # ------------------------------------------------------------------------------------------------------------------
    img_path_bbox_attr_cls_tuples_list = []
    # with open(fashion_dataset_path + 'deepfashion.txt') as f:
    #     for line in f:
    #         line = line.split()
    #         if line[1] == 'None' or line[2] == 'None':
    #             continue
    #         attrs = []
    #         for a in map(int, line[2].split('-')):
    #             if F[a] in attr300:
    #                 attrs.append(str(attr300.index(F[a])))
    #         if not attrs:
    #             continue
    #         img_path_bbox_attr_cls_tuples_list.append((line[0], line[1], '-'.join(attrs), line[3], line[4], line[5]))
    sz = 0
    with open(fashion_dataset_path + 'augmented_350_2420.txt') as f:
        for line in f:
            line = line.split()
            img_path_bbox_attr_cls_tuples_list.append((line[0], line[1], line[2], line[3], line[4], line[5]))
            sz += 1
    shuffle(img_path_bbox_attr_cls_tuples_list)
    # shuffle(img_path_bbox_attr_cls_tuples_list)
    # neg_class_size = int(sz*0.01)
    # count = 0
    # if not os.path.exists(fashion_dataset_path + 'augmented/neg_class'):
    #     os.mkdir(fashion_dataset_path + 'augmented/neg_class')
    # for img_path, img_gt_bbox, img_attr, img_class, w, h in img_path_bbox_attr_cls_tuples_list:
    #     if img_class != 'None':
    #         continue
    #     save_path = fashion_dataset_path + 'augmented/neg_class/img'+str(count).zfill(8)+'.jpg'
    #     shutil.copy(img_path, save_path)
    #     img_path_bbox_attr_cls_tuples_list.append((save_path, img_gt_bbox, img_attr, img_class, w, h))
    #     count += 1
    #     if count >= neg_class_size:
    #         break
    # shuffle(img_path_bbox_attr_cls_tuples_list)
    partition = 0.95
    with open(os.path.join(fashion_dataset_path, 'train_95-5.txt'), 'w') as f_train:
        for i in range(int(len(img_path_bbox_attr_cls_tuples_list) * partition)):
            img_path, img_gt_bbox, img_attr, img_class, width, height = img_path_bbox_attr_cls_tuples_list[i]
            f_train.write('{} {} {} {} {} {}\n'.format(img_path, img_gt_bbox, img_attr, img_class, width, height))
    with open(os.path.join(fashion_dataset_path, 'validation_95-5.txt'), 'w') as f_validation:
        for i in range(i, len(img_path_bbox_attr_cls_tuples_list)):
            img_path, img_gt_bbox, img_attr, img_class, width, height = img_path_bbox_attr_cls_tuples_list[i]
            f_validation.write(
                '{} {} {} {} {} {}\n'.format(img_path, img_gt_bbox, img_attr, img_class, width, height))
    # ---------------------------------------------------------------------------------------------------------
    # attr_labels = []
    # class_labels = []
    # for i in range(len(img_path_bbox_attr_cls_tuples_list)):
    #     line = img_path_bbox_attr_cls_tuples_list[i][2]
    #     if line.split('-')[0] != 'None':
    #         for y in list(map(int, line.split('-'))):
    #                 attr_labels.append(y)
    #     if img_path_bbox_attr_cls_tuples_list[i][3] != 'None':
    #         class_labels.append(img_path_bbox_attr_cls_tuples_list[i][3])
    # cls_names = np.unique(class_labels)
    # cls_weight = class_weight.compute_class_weight('balanced', cls_names, class_labels)
    # print(cls_names)
    # for i in range(len(cls_names)):
    #     print(cls_names[i], len([x for x in class_labels if x == cls_names[i]]), cls_weight[i])
    # with open(fashion_dataset_path + 'class_weights.pkl', 'wb') as f:
    #     pickle.dump(cls_weight, f)
    # attr_weight = class_weight.compute_class_weight('balanced', range(1000), attr_labels)
    # for i in range(1000):
    #     print(i, attr_weight[i])
    # with open(fashion_dataset_path + 'attr_weights.pkl', 'wb') as f:
    #     pickle.dump(attr_weight, f)
