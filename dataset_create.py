import binascii
import os
import pickle
import numpy as np
import glob
import fnmatch
from random import shuffle, choice
import shutil
import logging

import skimage
from sklearn.utils import class_weight

logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")
from PIL import Image
from utils import init_globals, bb_intersection_over_union
from segmentation import cluster_bboxes, selective_search_aggregated
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import apply_affine_transform, flip_axis

fashion_dataset_path = 'fashion_data/'


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


def calculate_bbox_score_and_save_img(image_path_name, image_save_path, bb_lst, cls_lst, fw, count):
    image = Image.open(image_path_name)
    w, h = image.size[0], image.size[1]
    new_w, new_h = w, h
    res_value = 400
    if max(w, h) > res_value:
        if w > h:
            new_w = res_value
            new_h = int(h / w * res_value)
        else:
            new_h = res_value
            new_w = int(w / h * res_value)
            image = image.resize((new_w, new_h))
    img_as_np_arr = np.array(image)
    if img_as_np_arr.shape[2] > 3:
        img_as_np_arr = image[:, :, :3]
    candidates = cluster_bboxes(selective_search_aggregated(img_as_np_arr), w, h, new_w, new_h, preference=-0.03,
                                fast=True)
    for x, y, w, h in candidates:
        cls_arr = np.zeros((3,))
        for bb, icl in zip(bb_lst, cls_lst):
            xc, yc = bb[0] + (bb[2] - bb[0]) / 2, bb[1] + (bb[3] - bb[1]) / 2
            if x < xc < x + w and y < yc < y + h:
                cls_arr[int(icl) - 1] = max(cls_arr[int(icl) - 1],
                                            bb_intersection_over_union((bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]),
                                                                       (x, y, w, h)))
        img_crop = image.crop((x, y, x + w, y + h))
        cls_string = '-'.join(map(str, cls_arr))
        image_save_name = cls_string + '_' + binascii.b2a_hex(os.urandom(5)).decode("utf-8")
        image_save_path_name = image_save_path + '/' + image_save_name + '.jpg'
        img_crop.save(image_save_path_name)
        fw.write('{} {}\n'.format(image_save_path_name, cls_string))
        count = count + 1
        print(count, image_save_path_name)
        # Ground Truth
    for gt_bbox, clss in zip(bb_lst, cls_lst):
        img_crop = image.crop(gt_bbox)
        cls_arr = np.zeros((3,))
        cls_arr[int(clss) - 1] = 1.
        cls_string = '-'.join(map(str, cls_arr))
        image_save_name = cls_string + '_' + binascii.b2a_hex(os.urandom(5)).decode("utf-8")
        image_save_path_name = image_save_path + '/' + image_save_name + '.jpg'
        img_crop.save(image_save_path_name)
        fw.write('{} {}\n'.format(image_save_path_name, cls_string))
        count = count + 1
        print(count, image_save_path_name)
    return count


def generate_dataset(img_bbox_cls, i):
    count = 0
    with open('cropped' + str(i) + '.txt', "w") as w:
        for img_path, image_save_path, bb_lst, cls_lst in img_bbox_cls:
            count = calculate_bbox_score_and_save_img(img_path, image_save_path, bb_lst, cls_lst, w, count)


def find_square_bbox(bbox, width, height):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    xc, yc = int(bbox[0] + w / 2), int(bbox[1] + h / 2)
    a = min(width, height, max(w, h, 224))
    x1, y1, x2, y2 = xc - a // 2, yc - a // 2, xc + a // 2, yc + a // 2
    if x1 < 0:
        x2 += -x1
        x1 += -x1
    if y1 < 0:
        y2 += -y1
        y1 += -y1
    if x2 > width:
        x1 -= x2 - width
        x2 -= x2 - width
    if y2 > height:
        y2 -= y2 - height
        y1 -= y2 - height
    return (x1, y1, x2, y2)


if __name__ == '__main__':
    global class_names, input_shape, attr_names, class_cloth_type, type_names, class35, attr200, attr300, F
    class_names, class_type, input_shape, attr_names = init_globals(fashion_dataset_path)
    attr491 = [706, 598, 487, 989, 313, 312, 579, 660, 828, 440, 209, 498, 755, 166, 972, 274, 954, 41, 391, 739, 792,
               882, 347, 346, 861, 915, 180, 939, 809, 85, 932, 117, 804, 157, 349, 548, 866, 485, 317, 418, 475, 535,
               789, 952, 992, 125, 178, 741, 74, 634, 40, 111, 973, 128, 232, 659, 113, 860, 465, 223, 308, 385, 585,
               76, 176, 235, 467, 91, 610, 434, 700, 834, 473, 888, 776, 405, 103, 258, 576, 442, 318, 334, 126, 798,
               678, 105, 127, 82, 510, 582, 505, 515, 890, 118, 767, 950, 275, 657, 372, 140, 452, 296, 155, 651, 192,
               528, 60, 783, 835, 478, 727, 566, 444, 965, 419, 526, 631, 734, 461, 703, 100, 234, 816, 521, 174, 614,
               629, 612, 710, 315, 251, 587, 109, 686, 77, 5, 238, 265, 545, 603, 712, 847, 597, 773, 433, 927, 606, 80,
               519, 407, 483, 757, 736, 469, 714, 609, 222, 291, 361, 893, 68, 746, 23, 642, 303, 397, 873, 658, 806,
               540, 84, 940, 411, 354, 520, 821, 293, 210, 284, 930, 17, 43, 687, 628, 967, 150, 921, 987, 827, 977,
               671, 812, 104, 393, 654, 39, 246, 476, 73, 132, 307, 119, 872, 203, 725, 277, 869, 72, 999, 273, 763,
               756, 359, 389, 449, 842, 114, 532, 777, 974, 446, 799, 146, 416, 110, 474, 936, 19, 848, 224, 202, 489,
               907, 669, 328, 396, 90, 283, 840, 279, 667, 183, 360, 544, 694, 696, 662, 917, 431, 302, 735, 941, 901,
               708, 124, 188, 891, 468, 948, 414, 272, 619, 24, 889, 482, 929, 324, 201, 567, 971, 902, 768, 131, 327,
               262, 701, 793, 569, 189, 871, 159, 154, 58, 947, 47, 943, 618, 723, 970, 683, 909, 960, 674, 27, 854,
               691, 153, 330, 953, 649, 561, 666, 931, 38, 20, 480, 958, 325, 538, 788, 784, 423, 920, 36, 716, 624, 42,
               18, 877, 785, 450, 25, 108, 562, 70, 218, 900, 774, 599, 116, 982, 531, 356, 30, 410, 648, 264, 802, 231,
               880, 306, 139, 682, 45, 512, 797, 184, 638, 775, 309, 911, 245, 689, 336, 129, 986, 87, 937, 392, 862,
               811, 852, 749, 48, 764, 15, 699, 887, 115, 321, 899, 208, 843, 653, 560, 543, 249, 988, 44, 747, 437,
               815, 287, 429, 851, 99, 998, 276, 608, 415, 841, 141, 786, 14, 83, 800, 754, 11, 676, 133, 368, 697, 191,
               984, 409, 413, 688, 438, 944, 897, 453, 300, 563, 237, 186, 207, 50, 574, 825, 593, 692, 751, 770, 565,
               571, 817, 881, 0, 993, 695, 823, 457, 204, 722, 839, 239, 713, 681, 61, 268, 698, 227, 93, 913, 282, 781,
               470, 236, 121, 337, 818, 878, 292, 616, 983, 933, 358, 969, 935, 820, 112, 601, 720, 831, 181, 956, 640,
               1, 577, 620, 830, 226, 81, 745, 705, 310, 568, 760, 717, 441, 162, 335, 546, 196, 353, 380, 892, 837,
               883, 212, 142]
    class33 = ['Bomber', 'Flannel', 'Button-Down', 'Trunks', 'Culottes', 'Chinos', 'Jeggings', 'Parka', 'Henley',
                   'Jersey', 'Poncho', 'Sweatshorts', 'Cutoffs', 'Coat', 'Kimono', 'Sweatpants', 'Hoodie', 'Joggers',
                   'Leggings', 'Jumpsuit', 'Jeans', 'Blazer', 'Romper', 'Top', 'Jacket', 'Sweater', 'Cardigan', 'Skirt',
                   'Tank', 'Shorts', 'Blouse', 'Tee', 'Dress']
    # --------------------DeepFashion dataset stats---------------------------------------------------------------------
    with open(fashion_dataset_path + 'deepfashion.txt', 'w') as f:
        with open(fashion_dataset_path + '/Anno/list_bbox.txt') as file_bbox:
            with open(fashion_dataset_path + '/Anno/list_category_img.txt') as file_category:
                with open(os.path.join(fashion_dataset_path, 'Eval/list_eval_partition.txt')) as file_partition:
                    with open(os.path.join(fashion_dataset_path, 'Anno/list_attr_img.txt')) as file_attr:
                        next(file_attr)
                        next(file_attr)
                        for line in file_attr:
                            line = line.split()
                            img_path = os.path.join(fashion_dataset_path, 'Img', line[0])
                            img = Image.open(img_path)
                            img_attr = np.array(eval('[' + ','.join(line[1:]) + ']'))
                            img_attr = list(np.argwhere(img_attr == 1))
                            img_attr_str = '-'.join(map(str, [x[0] for x in img_attr]))
                            if len(img_attr) == 0:
                                continue
                            img_class = class_names[int(get_second_arg_from_file(line[0], file_category)) - 1]
                            img_gt_bbox = get_gt_bbox_from_file(line[0], file_bbox)
                            img_gt_bbox_str = '-'.join(map(str, img_gt_bbox))
                            img_part = get_second_arg_from_file(line[0], file_partition)
                            f.write('{} {} {} {} {}\n'.format(img_path, img_gt_bbox_str, img_attr_str, img_class, img_part))
    # ------------------------------------------------------------------------------------------------------------------
    classN = set(class33)
    img_path_bbox_attr_cls_tuples_list = []
    with open(fashion_dataset_path + 'deepfashion.txt') as f:
        for line in f:
            line = line.split()
            if line[1] == 'None' or line[2] == 'None':
                continue
            if line[3] in classN:
                img_path_bbox_attr_cls_tuples_list.append((line[0], line[1], line[2], line[3], line[4]))

    # ------------------------------------------------------------------------------------------------------------------
    attrN = attr491
    attrNset = set(attrN)
    crop = 5
    attr_count = dict(((x, []) for x in range(len(attrN))))
    for i in range(len(img_path_bbox_attr_cls_tuples_list)):
        line = img_path_bbox_attr_cls_tuples_list[i][2]
        for y in list(map(int, line.split('-'))):
            if y in attrNset:
                attr_count[attrN.index(y)].append(i)
    attr_count_list = sorted([(len(attr_count[i]), i) for i in range(len(attrN))])
    img_gen = ImageDataGenerator(rotation_range=30,
                                 horizontal_flip=True,
                                 width_shift_range=0.15,
                                 height_shift_range=0.15,
                                 shear_range=0.1,
                                 zoom_range=0.1)
    attr_count_crop = dict(((x, 0) for x in range(len(attrN))))
    cls_count = dict(((x, 0) for x in class_names))
    registred = set()
    if not os.path.exists(fashion_dataset_path + 'augmented'):
        os.mkdir(fashion_dataset_path + 'augmented')
    with open(fashion_dataset_path + 'augmented_' + str(len(attrN)) + '_' + str(crop) + '.txt', 'w') as f:
        for value, key in attr_count_list:
            dir = str(key)
            if not os.path.exists(fashion_dataset_path + 'augmented/' + dir):
                os.mkdir(fashion_dataset_path + 'augmented/' + dir)
            attr_count[key] = list(set(attr_count[key]) - registred)
            shuffle(attr_count[key])
            attr_count[key] = attr_count[key][:crop]
            unique = []
            for i in attr_count[key]:
                save_path = fashion_dataset_path + 'augmented/' + dir + '/img' + str(attr_count_crop[key]).zfill(
                    8) + '.jpg'
                line = list(map(int, img_path_bbox_attr_cls_tuples_list[i][2].split('-')))
                attr_list = [attrN.index(x) for x in line if x in attrN]
                image = Image.open(img_path_bbox_attr_cls_tuples_list[i][0].replace('\\', '/'))
                w, h = image.size[0], image.size[1]
                bbox = [int(x) for x in img_path_bbox_attr_cls_tuples_list[i][1].split('-')]
                image = image.crop(find_square_bbox(bbox, w, h))
                image.save(save_path)
                f.write('{} {} {}\n'.format(save_path, '-'.join(map(str, attr_list)),
                                            img_path_bbox_attr_cls_tuples_list[i][3]))
                for y in attr_list:
                    attr_count_crop[y] += 1
                cls_count[img_path_bbox_attr_cls_tuples_list[i][3]] += 1
                registred.add(i)
                unique.append(i)

            while attr_count_crop[key] < crop:
                ii = choice(unique)
                arr = np.array(skimage.io.imread(img_path_bbox_attr_cls_tuples_list[ii][0].replace('\\', '/')))
                w, h = arr.shape[1], arr.shape[0]
                bbox = [float(x) for x in img_path_bbox_attr_cls_tuples_list[ii][1].split('-')]
                if len(arr.shape) != 3 or arr.shape[2] != 3:
                    continue
                transform_parameters = img_gen.get_random_transform(arr.shape, seed=attr_count_crop[key])
                res = img_gen.random_transform(arr, seed=attr_count_crop[key])
                x = np.zeros((h, w, 3))
                x[int(bbox[1] * h):int(bbox[3] * h) - 1, int(bbox[0] * w):int(bbox[2] * w) - 1, 0] = 100
                x = apply_affine_transform(x, transform_parameters.get('theta', 0),
                                           transform_parameters.get('tx', 0),
                                           transform_parameters.get('ty', 0),
                                           transform_parameters.get('shear', 0),
                                           transform_parameters.get('zx', 1),
                                           transform_parameters.get('zy', 1),
                                           row_axis=0, col_axis=1,
                                           channel_axis=2,
                                           fill_mode='nearest', cval=0.)
                if transform_parameters.get('flip_horizontal', False):
                    x = flip_axis(x, 1)
                if transform_parameters.get('flip_vertical', False):
                    x = flip_axis(x, 0)
                x = x[:, :, 0]
                arr_h = np.max(x, axis=1);
                arr_w = np.max(x, axis=0)
                i = 0;
                while (i < h and arr_h[i] < 1e-14):
                    i += 1
                y1 = i;
                i = h - 1;
                while (i >= 0 and arr_h[i] < 1e-14):
                    i -= 1
                y2 = i;
                i = 0;
                while (i < w and arr_w[i] < 1e-14):
                    i += 1
                x1 = i;
                i = w - 1;
                while (i >= 0 and arr_w[i] < 1e-14):
                    i -= 1
                x2 = i;
                bbox = (x1, y1, x2, y2)
                line = list(map(int, img_path_bbox_attr_cls_tuples_list[ii][2].split('-')))
                attr_list = [attrN.index(x) for x in line if x in attrN]
                cls = img_path_bbox_attr_cls_tuples_list[ii][3]
                path = fashion_dataset_path + 'augmented/' + dir + '/img' + str(attr_count_crop[key]).zfill(8) + '.jpg'
                for y in attr_list:
                    attr_count_crop[y] += 1
                im = Image.fromarray(res)
                im = im.crop(find_square_bbox(bbox, w, h))
                im.save(path)
                f.write('{} {} {}\n'.format(path, '-'.join(map(str, attr_list)), cls))
    aaa = sorted([(attr_count_crop[i], i) for i in range(len(attrN))])
    bbb = sorted([(cls_count[i], i) for i in class_names])
    for val, key in aaa:
        print(key, val)
    for val, key in bbb:
        print(key, val)
    # ------------------------------------------------------------------------------------------------------------------
    img_path_bbox_attr_cls_tuples_list = []
    with open(fashion_dataset_path + 'augmented_' + str(len(attrN)) + '_' + str(crop) + '.txt') as f:
        for line in f:
            line = line.split()
            img_path_bbox_attr_cls_tuples_list.append((line[0], line[1], line[2]))
    shuffle(img_path_bbox_attr_cls_tuples_list)
    partition = 0.95
    with open(os.path.join(fashion_dataset_path, 'train_95-5.txt'), 'w') as f_train:
        for i in range(int(len(img_path_bbox_attr_cls_tuples_list) * partition)):
            img_path, img_attr, img_cls = img_path_bbox_attr_cls_tuples_list[i]
            f_train.write('{} {} {}\n'.format(img_path, img_attr, img_cls))
    with open(os.path.join(fashion_dataset_path, 'validation_95-5.txt'), 'w') as f_validation:
        for i in range(i, len(img_path_bbox_attr_cls_tuples_list)):
            img_path, img_attr, img_cls = img_path_bbox_attr_cls_tuples_list[i]
            f_validation.write('{} {} {}\n'.format(img_path, img_attr, img_cls))
