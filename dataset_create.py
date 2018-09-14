import os
import pickle
import numpy as np
import glob
import fnmatch
from random import randint, shuffle
import shutil
import logging
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")
from PIL import Image
from utils import init_globals, bb_intersection_over_union
from segmentation import selective_search_bbox_fast, cluster_bboxes
from multiprocessing import Pool

# GLOBALS
num_proc = 4
# dataset_path='../Data/dataset_3heads'
# dataset_train_path=os.path.join(dataset_path, 'train')
# dataset_val_path=os.path.join(dataset_path, 'validation')
# dataset_test_path=os.path.join(dataset_path, 'test')
fashion_dataset_path = 'fashion_data/'

def create_category_structure(category_names):
    for category_name in category_names:
        # Train
        category_path_name = os.path.join(dataset_train_path, category_name)
        if not os.path.exists(os.path.join(category_path_name)):
            os.makedirs(category_path_name)
        # Validation
        category_path_name = os.path.join(dataset_val_path, category_name)
        if not os.path.exists(os.path.join(category_path_name)):
            os.makedirs(category_path_name)
        # Test
        category_path_name = os.path.join(dataset_test_path, category_name)
        if not os.path.exists(os.path.join(category_path_name)):
            os.makedirs(category_path_name)


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


def generate_dataset_gt_bbox_crop():
    count = -1
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
                        img_part = get_second_arg_from_file(img_path, file_partition)
                        if img_part == 'val':
                            img_part = 'validation'
                        img_class_indx = get_second_arg_from_file(img_path, file_category)
                        img_class = class_names[int(img_class_indx) - 1]
                        img_gt_bbox = get_gt_bbox_from_file(img_path, file_bbox)
                        logging.debug('Reading {} {}'.format(img_path, img_gt_bbox))
                        image_save_name = img_path.split('/')[-2] + '_' + img_path.split('/')[-1].split('.')[0]
                        attrs_str = '-'.join(map(str, [x[0] for x in list(np.argwhere(img_attr == 1))]))
                        image_save_path = os.path.join(dataset_path, img_part, img_class)
                        image_save_path_name = image_save_path + '/' + image_save_name + '_' + attrs_str + '.jpg'
                        image = Image.open(os.path.join(fashion_dataset_path, 'Img', img_path))
                        img_crop = image.crop((img_gt_bbox[0], img_gt_bbox[1], img_gt_bbox[2], img_gt_bbox[3]))
                        img_crop.save(image_save_path_name)
                        logging.info('{} - {}'.format(count, image_save_path_name))


def select_subset_images_iou(crop_val):
    classes_idx = dict(((x, []) for x in class_names))
    count = -1
    with open(fashion_dataset_path + '/Anno/list_bbox.txt') as file_bbox:
        with open(fashion_dataset_path + '/Anno/list_category_img.txt') as file_category:
            next(file_category)
            next(file_category)
            for line in file_category:
                count += 1
                line = line.split()
                img_path = line[0]
                img_class = class_names[int(line[1]) - 1]
                img_gt_bbox = get_gt_bbox_from_file(img_path, file_bbox)
                img_bbox_size = (img_gt_bbox[2] - img_gt_bbox[0]) * (img_gt_bbox[3] - img_gt_bbox[1])
                classes_idx[img_class].append((img_bbox_size, count))
    for x in class_names:
        classes_idx[x] = sorted(classes_idx[x])[:crop_val]
    return [x[1] for y in class_names for x in classes_idx[y]]


def select_subset_images_3heads(crop_train, crop_val):
    attrs_classes_parts_tuples = []
    count = -1
    with open(fashion_dataset_path + '/Anno/list_category_img.txt') as file_category:
        with open(os.path.join(fashion_dataset_path, 'Eval/list_eval_partition.txt')) as file_partition:
            with open(os.path.join(fashion_dataset_path, 'Anno/list_attr_img.txt')) as file_attr:
                next(file_attr)
                next(file_attr)
                for line in file_attr:
                    count += 1
                    line = line.split()
                    img_path = line[0]
                    img_class_indx = get_second_arg_from_file(img_path, file_category)
                    img_class = class_names[int(img_class_indx) - 1]
                    img_attr = np.array(eval('[' + ','.join(line[1:]) + ']'))
                    img_part = get_second_arg_from_file(img_path, file_partition)
                    img_attrs = [x[0] for x in list(np.argwhere(img_attr == 1))]
                    attrs_classes_parts_tuples.append((img_attrs, img_class, img_part, count))
    shuffle(attrs_classes_parts_tuples)
    ret = dict()
    for part in ['train', 'val']:
        crop = eval('crop_' + part)
        classes_idx = dict(((x, set()) for x in classes35))
        attrs_idx = dict(((x, set()) for x in attrs200))
        ret[part] = set()

        for img_attrs, img_class, img_part, count in attrs_classes_parts_tuples:
            if img_part != part:
                continue
            skip = True
            for x in img_attrs:
                if x in attrs_idx.keys() and len(attrs_idx[x]) < crop:
                    skip = False
                    break
            if skip:
                continue
            for x in img_attrs:
                if x in attrs_idx.keys():
                    attrs_idx[x].add(count)
            if img_class in classes_idx.keys():
                classes_idx[img_class].add(count)
            if np.min([len(x) for x in attrs_idx.values()]) == crop:
                break

        for img_attrs, img_class, img_part, count in attrs_classes_parts_tuples:
            if img_part != part:
                continue
            if img_class not in classes_idx.keys() or len(classes_idx[img_class]) >= crop:
                continue
            classes_idx[img_class].add(count)
            if np.min([len(x) for x in classes_idx.values()]) == crop:
                break

        for x in classes35:
            ret[part] = ret[part] | classes_idx[x]
        for x in attrs200:
            ret[part] = ret[part] | attrs_idx[x]
        for x in classes35:
            logging.info('num samples for {}: {}'.format(x, len(ret[part] & classes_idx[x])))
        for x in attrs200:
            logging.info('num samples for {}: {}'.format(x, len(ret[part] & attrs_idx[x])))
        logging.info('min samples for attr: {}'.format(np.min([len(x) for x in attrs_idx.values()])))
        logging.info('max samples for attr: {}'.format(np.max([len(x) for x in attrs_idx.values()])))
        logging.info('min samples for class: {}'.format(np.min([len(x) for x in classes_idx.values()])))
        logging.info('max samples for class: {}'.format(np.max([len(x) for x in classes_idx.values()])))
        logging.info('{} samples: {}'.format(part, str(len(ret[part]))))
        with open(os.path.join(dataset_path, part + '.txt'), 'w') as f:
            for x in ret[part]:
                f.write(str(x) + '\n')
    return (ret['train'], ret['val'])


def generate_dataset_iou(indexes):
    count = -1
    with open(fashion_dataset_path + '/Anno/list_bbox.txt') as file_bbox:
        with open(fashion_dataset_path + '/Anno/list_category_img.txt') as file_category:
            with open(os.path.join(fashion_dataset_path, 'Eval/list_eval_partition.txt')) as file_partition:
                next(file_category)
                next(file_category)
                for line in file_category:
                    count += 1
                    if count not in indexes:
                        continue
                    line = line.split()
                    img_path = line[0]
                    img_class = class_names[int(line[1]) - 1]
                    img_part = get_second_arg_from_file(img_path, file_partition)
                    if img_part == 'val':
                        img_part = 'validation'
                    img_gt_bbox = get_gt_bbox_from_file(img_path, file_bbox)
                    img_type = type_names[class_cloth_type[img_class]]
                    image_save_path = os.path.join(dataset_path, img_part, img_type)
                    calculate_bbox_score_and_save_img(os.path.join(fashion_dataset_path, 'Img', img_path),
                                                      image_save_path, img_gt_bbox)
                    logging.info('{} - {}'.format(count, image_save_path))


if __name__ == '__main__':
    global class_names, input_shape, attr_names, class_cloth_type, type_names, class35, attr200
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
    # generate_dataset_gt_bbox_crop()
    # create_category_structure(type_names)
    # indexes = select_subset_images_iou(300)
    # jobs = []
    # b = len(indexes) // num_proc
    # for i in range(num_proc - 1):
    #     jobs.append(indexes[i*b:(i+1)*b])
    # jobs.append(indexes[(num_proc - 1)*b:])
    # p = Pool(num_proc)
    # p.map(generate_dataset_iou, jobs)

    # ---------------------3 HEADS WHOLE--------------------------
    # create_category_structure(class_names)
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

    # ---------------------3 HEADS CROPPED------------------------------
    # select_subset_images_3heads(100, 10)
    # indexes = []
    # with open(dataset_path+'/train.txt') as f:
    #     for l in f:
    #         indexes.append(int(l))
    # with open(dataset_path + '/val.txt') as f:
    #     for l in f:
    #         indexes.append(int(l))
    # jobs = []
    # b = len(indexes) // num_proc
    # for i in range(num_proc - 1):
    #     jobs.append(indexes[i*b:(i+1)*b])
    # jobs.append(indexes[(num_proc - 1)*b:])
    # p = Pool(num_proc)
    # p.map(generate_dataset_three_heads_from_indexes, jobs)
    # # generate_dataset_three_heads((0,total_count))

    # attr_stat = np.zeros((1000,))
    # with open(os.path.join(fashion_dataset_path, 'Anno/list_attr_img.txt')) as f:
    #     next(f)
    #     next(f)
    #     for line in f:
    #         line = line.split()
    #         attrs = eval('[' + ','.join(line[1:]) + ']')
    #         for i,a in enumerate(attrs):
    #             if a == 1:
    #                 attr_stat[i] += 1
    # amounts = [(attr_stat[i], i) for i in range(1000)]
    # b=sorted(amounts, reverse=True)
    # print([x[1] for x in b[:200]])

    # --------------------DeepFashion dataset stats---------------------
    # img_neg_class = []
    # for root, dirnames, filenames in sorted(os.walk('../Data/fashion_data/Img/img/neg_class')):
    #     for filename in sorted(fnmatch.filter(filenames, '*.jpg')):
    #         full_path = os.path.join(root, filename)
    #         img = Image.open(full_path)
    #         width, height = img.size[0], img.size[1]
    #         img_neg_class.append((full_path, '0.0-0.0-0.0-0.0', 'None', 'Background', width, height))

    with open(fashion_dataset_path + 'train85.txt', 'w') as f_train:
        with open(fashion_dataset_path + 'validation85.txt', 'w') as f_validation:
            # with open(fashion_dataset_path + 'test.txt', 'w') as f_test:
            img_path_bbox_attr_cls_tuples_list = []
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
                                width, height = img.size[0], img.size[1]
                                img_attr = np.array(eval('[' + ','.join(line[1:]) + ']'))
                                img_attr = '-'.join(map(str, [x[0] for x in list(np.argwhere(img_attr == 1))]))
                                if len(img_attr) == 0:
                                    img_attr = None
                                img_class = class_names[int(get_second_arg_from_file(line[0], file_category)) - 1]
                                img_gt_bbox = get_gt_bbox_from_file(line[0], file_bbox)
                                img_gt_bbox = [img_gt_bbox[0] / width, img_gt_bbox[1] / height, img_gt_bbox[2] / width,
                                               img_gt_bbox[3] / height]
                                img_gt_bbox = '-'.join(map(str, img_gt_bbox))
                                img_part = get_second_arg_from_file(line[0], file_partition)
                                if img_part == 'val':
                                    img_part = 'validation'
                                # file_to_write = eval('f_{}'.format(img_part))
                                # file_to_write.write('{} {} {} {}\n'.format(img_path, img_gt_bbox, img_attr, img_class))
                                img_path_bbox_attr_cls_tuples_list.append(
                                    (img_path, img_gt_bbox, img_attr, img_class, width, height))
            with open(fashion_dataset_path + 'train85.txt') as f:
                for line in f:
                    line = line.split()
                    img_path_bbox_attr_cls_tuples_list.append((line[0], line[1], line[2], line[3], line[4], line[5]))
            with open(fashion_dataset_path + 'validation85.txt') as f:
                for line in f:
                    line = line.split()
                    img_path_bbox_attr_cls_tuples_list.append((line[0], line[1], line[2], line[3], line[4], line[5]))
            # img_path_bbox_attr_cls_tuples_list = img_path_bbox_attr_cls_tuples_list + img_neg_class
            shuffle(img_path_bbox_attr_cls_tuples_list)
            shuffle(img_path_bbox_attr_cls_tuples_list)
            shuffle(img_path_bbox_attr_cls_tuples_list)
            for i in range(int(len(img_path_bbox_attr_cls_tuples_list) * 0.85)):
                img_path, img_gt_bbox, img_attr, img_class, width, height = img_path_bbox_attr_cls_tuples_list[i]
                f_train.write('{} {} {} {} {} {}\n'.format(img_path, img_gt_bbox, img_attr, img_class, width, height))
            for i in range(i, len(img_path_bbox_attr_cls_tuples_list)):
                img_path, img_gt_bbox, img_attr, img_class, width, height = img_path_bbox_attr_cls_tuples_list[i]
                f_validation.write(
                    '{} {} {} {} {} {}\n'.format(img_path, img_gt_bbox, img_attr, img_class, width, height))
    # -------------------------------------------------
    new_class_weights = []
    new_attr_weights = []
    with open(fashion_dataset_path + 'train85.txt') as f:
        for line in f:
            line = line.split()
            if line[3] in class35:
                new_class_weights.append(line[3])
            if line[2].split('-')[0] != 'None':
                for y in list(map(int, line[2].split('-'))):
                    if y in attr200:
                        new_attr_weights.append(y)
    with open(fashion_dataset_path + 'validation85.txt') as f:
        for line in f:
            line = line.split()
            if line[3] in class35:
                new_class_weights.append(line[3])
            if line[2].split('-')[0] != 'None':
                for y in list(map(int, line[2].split('-'))):
                    if y in attr200:
                        new_attr_weights.append(y)
    with open(fashion_dataset_path + 'class_data_train85.pkl', 'wb') as f:
        pickle.dump(new_class_weights, f)
    with open(fashion_dataset_path + 'attr_data_train85.pkl', 'wb') as f:
        pickle.dump(new_attr_weights, f)
    pass
