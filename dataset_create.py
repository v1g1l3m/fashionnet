import os
import numpy as np
import glob
import fnmatch
from random import randint, shuffle
import shutil
import logging
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")
import skimage.data
from skimage import exposure
import PIL
from PIL import Image
from utils import init_globals, bb_intersection_over_union
from segmentation import selective_search_clustered
from multiprocessing import Pool

#GLOBALS
num_proc = 4
dataset_path='../Data/dataset_3heads'
dataset_train_path=os.path.join(dataset_path, 'train')
dataset_val_path=os.path.join(dataset_path, 'validation')
dataset_test_path=os.path.join(dataset_path, 'test')
fashion_dataset_path='../Data/fashion_data/'

def create_category_structure(category_names):
    for category_name in category_names:
        # Train
        category_path_name=os.path.join(dataset_train_path, category_name)
        if not os.path.exists(os.path.join(category_path_name)):
            os.makedirs(category_path_name)
        # Validation
        category_path_name=os.path.join(dataset_val_path, category_name)
        if not os.path.exists(os.path.join(category_path_name)):
            os.makedirs(category_path_name)
        # Test
        category_path_name=os.path.join(dataset_test_path, category_name)
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
            x1=int(line.split()[1])
            y1=int(line.split()[2])
            x2=int(line.split()[3])
            y2=int(line.split()[4])
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
    candidates = selective_search_clustered(image_path_name)
    img_read = Image.open(image_path_name)
    for x, y, w, h in (candidates):
        boxA = gt_bbox
        boxB = (x, y, x+w, y+h)
        iou = bb_intersection_over_union(boxA, boxB)
        img_crop = img_read.crop((x, y, x+w, y+h))
        image_save_name = image_path_name.split('/')[-2] + '_' + image_path_name.split('/')[-1].split('.')[0] + '_' + img_type + '_' + attrs_str + '_'
        image_save_path_name = image_save_path + '/' + image_save_name + str(iou) + '.jpg'
        logging.debug('image_save_path_name {}'.format(image_save_path_name))
        non_blocking_save(img_crop, image_save_path_name)
    # Ground Truth
    img_crop = img_read.crop(gt_bbox)
    image_save_name = image_path_name.split('/')[-2] + '_' + image_path_name.split('/')[-1].split('.')[0] + '_' + img_type + '_' + attrs_str + '_'
    image_save_path_name = image_save_path + '/' + image_save_name + '1.0' + '.jpg'
    logging.debug('image_save_path_name {}'.format(image_save_path_name))
    non_blocking_save(img_crop, image_save_path_name)

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
                                                          img_type_indx + '_' + attrs_str + '_'+'.jpg'))

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
                img_bbox_size = (img_gt_bbox[2] - img_gt_bbox[0])*(img_gt_bbox[3] - img_gt_bbox[1])
                classes_idx[img_class].append((img_bbox_size, count))
    for x in class_names:
        classes_idx[x] = sorted(classes_idx[x])[:crop_val]
    return [x[1] for y in class_names for x in classes_idx[y]]

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
                        calculate_bbox_score_and_save_img(os.path.join(fashion_dataset_path, 'Img', img_path), image_save_path, img_gt_bbox)
                        logging.info('{} - {}'.format(count, image_save_path))

global class_names, input_shape, attr_names, class_cloth_type, type_names
class_names, input_shape, attr_names = init_globals()
type_names = ['upper-body', 'lower-body', 'full-body']
class_cloth_type = dict()
with open(fashion_dataset_path + 'Anno/list_category_cloth.txt') as f:
    next(f)
    next(f)
    for line in f:
        line = line.split()
        class_cloth_type[line[0]] = int(line[1]) - 1

if __name__ == '__main__':
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
    create_category_structure(class_names)
    with open(fashion_dataset_path + 'Anno/list_category_img.txt') as f:
        total_count = int(f.readline())
    b = total_count // num_proc
    jobs = [((num_proc-1)*b, total_count)]
    for i in range(num_proc - 1):
        jobs.append((i*b, (i+1)*b))
    print(jobs)
    # p = Pool(num_proc)
    # p.map(generate_dataset_three_heads, jobs)
    generate_dataset_three_heads((0,total_count))
