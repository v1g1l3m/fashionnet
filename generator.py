import numpy as np
import os
import glob
import time

from keras.applications.vgg16 import preprocess_input
from skimage.io import imread
from PIL import Image
from random import shuffle
from multiprocessing import Process, Queue
from utils import  bb_intersection_over_union
import time

# GLOBALS
img_width = 224             # For VGG16
img_height = 224            # For VGG16
img_channel = 3
fashion_dataset_path = '../Data/fashion_data/'

def generate_arrays_from_file(path, batch_size, class_names, resize):
    images = []
    class_names_ = np.zeros((batch_size, len(class_names)), dtype=np.float32)
    iou_values = np.zeros((batch_size,), dtype=np.float32)
    while True:
        with open(path) as f:
            for image_path in f:
                current_index = len(images)
                img = Image.open(image_path.rstrip())
                img = img.resize(resize)
                img = np.array(img).astype(np.float32)
                img[:, :, 0] -= 103.939
                img[:, :, 1] -= 116.779
                img[:, :, 2] -= 123.68
                images.append(img)
                if os.name == 'nt':
                    image_path = image_path.replace('\\', '/')
                iou_values[current_index] = np.float(image_path.split('_')[-1].split('.jpg')[0])
                class_names_[current_index, class_names.index(image_path.split('/')[-2])] = iou_values[current_index]
                if current_index < batch_size - 1:
                    continue
                out = np.array(images)
                images = []
                yield (out, [class_names_, iou_values])

def image_read_transformer(dir, batch_size):
    class_names = []
    for name in sorted(os.listdir(dir)):
        if os.path.isdir(os.path.join(dir, name)):
            class_names.append(name)
    img_name_class_iou_tuples = []
    for class_name in class_names:
        dataset_path = os.path.join(dir, class_name)
        images_path_name = sorted(glob.glob(dataset_path + '/*.jpg'))
        for name in images_path_name:
            if os.name == 'nt':
                name = name.replace('\\', '/')
            iou = np.float(name.split('_')[-1].split('.jpg')[0])
            class_1_hot = np.zeros((len(class_names),), dtype=np.float32)
            class_1_hot[class_names.index(name.split('/')[-2])] = iou
            img_name_class_iou_tuples.append((name, class_1_hot, iou))
    while True:
        shuffle(img_name_class_iou_tuples)
        images_list = []
        class_1_hot_list = []
        iou_list = []
        for path, class_1_hot, iou in img_name_class_iou_tuples:
            current_batch_size = len(images_list)
            img = Image.open(path)
            img = img.resize((img_width, img_height))
            img = np.array(img).astype(np.float32)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
            images_list.append(img)
            class_1_hot_list.append(class_1_hot)
            iou_list.append(iou)
            if current_batch_size < batch_size - 1:
                continue
            out_img = np.array(images_list)
            out_cls = np.array(class_1_hot_list)
            out_iou = np.array(iou_list)
            images_list = []
            class_1_hot_list = []
            iou_list = []
            yield (out_img, [out_cls, out_iou])


def np_arrays_reader(path):
    while True:
        with open(path) as f:
            for btl_name in f:
                temp = np.load(open(btl_name.rstrip(), 'rb'))
                yield (temp['btl'], [temp['cls'], temp['iou']])


class Parallel_np_arrays_reader(object):

    def __init__(self, path, out_keys, maxsize=100, numproc=1):
        self.q = Queue(maxsize)
        self.path = os.path.split(path)[0]
        self.out_keys = out_keys
        self.np_arrays_path_list = []
        with open(path) as f:
            for btl_name in f:
                self.np_arrays_path_list.append(btl_name.rstrip())
        self.prs = []
        for i in range(numproc):
            p = Process(target=self.write_to_queue)
            p.start()
            self.prs.append(p)

    def __iter__(self):
        return self

    def write_to_queue(self):
        while True:
            shuffle(self.np_arrays_path_list)
            for btl_name in self.np_arrays_path_list:
                temp = np.load(open(os.path.join(self.path, btl_name), 'rb'))
                if len(self.out_keys) == 1:
                    out = temp[self.out_keys[0]]
                else:
                    out= []
                    for key in self.out_keys:
                        if key == 'bbiou':
                            out.append(temp[key][:,:4])
                        else:
                            out.append(temp[key])
                self.q.put((temp['btl'], out))
                
    def next(self):
        return self.q.get()

    def terminate(self):
        for p in self.prs:
            p.terminate()
            time.sleep(0.1)
            if not p.is_alive():
                p.join(timeout=1.0)
        self.q.close()

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    # with statement interface
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

class Parallel_image_read_transformer(object):

    def __init__(self, path, batch_size, class35, attr200, maxzise=100):
        self.batch_size = batch_size
        self.q = Queue(maxzise)
        self.img_name_class_attr_bbox_iou = []
        with open(path) as f:
            for line in f:
                line = line.split()
                img_path = line[0]
                img_bbox = [float(x) for x in line[1].split('-')]
                attrs_1_hot = np.zeros(200, )
                if line[2].split('-')[0] != 'None':
                    for x in line[2].split('-'):
                        if int(x) in attr200:
                            attrs_1_hot[attr200.index(int(x))] = 1
                class_1_hot = np.zeros((len(class35),), dtype=np.float32)
                if line[3] in class35:
                    class_1_hot[class35.index(line[3])] = 1
                w, h = int(line[4]), int(line[5])
                iou = bb_intersection_over_union([0, 0, w, h], [img_bbox[0] * w, img_bbox[1] * h, (img_bbox[2] - img_bbox[0]) * w, (img_bbox[3] - img_bbox[1]) * h])
                img_bbox.append(iou)
                self.img_name_class_attr_bbox_iou.append((img_path, class_1_hot, attrs_1_hot, img_bbox))
        self.p = Process(target=self.write_to_queue)
        self.p.start()

    def __iter__(self):
        return self

    def write_to_queue(self):
        while True:
            shuffle(self.img_name_class_attr_bbox_iou)
            images_list = []
            class_1_hot_list = []
            attrs_list = []
            bbox_list = []
            for path, class_1_hot, attrs_1_hot, bbox in self.img_name_class_attr_bbox_iou:
                current_batch_size = len(images_list)
                try:
                    img = Image.open(path)
                except:
                    print('Image not found:', path)
                    continue
                img = img.resize((img_width, img_height))
                img = np.array(img).astype(np.float32)
                images_list.append(img)
                class_1_hot_list.append(class_1_hot)
                attrs_list.append(attrs_1_hot)
                bbox_list.append(bbox)
                if current_batch_size < self.batch_size - 1:
                    continue
                out_img = np.array(images_list)
                out_img = preprocess_input(out_img)
                out_cls = np.array(class_1_hot_list)
                out_bbiou = np.array(bbox_list)
                out_attr = np.array(attrs_list)
                images_list = []
                class_1_hot_list = []
                attrs_list = []
                bbox_list = []
                self.q.put((out_img, out_attr))

    def next(self):
        return self.q.get()

    def terminate(self):
        self.p.terminate()
        time.sleep(0.1)
        if not self.p.is_alive():
            self.p.join(timeout=1.0)
            self.q.close()

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    # with statement interface
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

if __name__ == "__main__":
    class36 = ['Blazer', 'Top', 'Dress', 'Chinos', 'Jersey', 'Cutoffs', 'Kimono', 'Cardigan', 'Jeggings', 'Button-Down',
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
    # np_arrays_path_list = []
    # with open('E:\\ML\\bottleneck_test\\btl_validation_npz.txt') as f:
        # for btl_name in f:
            # np_arrays_path_list.append(btl_name.rstrip())
    # t1 = time.time()
    # for btl_name in np_arrays_path_list:
        # temp = np.load(open(btl_name, 'rb'))
    # print('total time of {} loads: {}'.format(len(np_arrays_path_list), time.time()-t1))
    # stop_value = 100
    # btl_path = 'bottleneck200/'
    # # train_generator = generate_arrays_from_file('bottleneck/btl_train.txt', 10, class_names, (224,224))
    # # train_generator = np_arrays_reader(btl_path+'/btl_train_npz.txt')
    # # tr_gen = image_transformer('../Data/dataset350/train/', 32)
    # # next(tr_gen)
    # tr_gen = image_read_transformer('../Data/dataset350/train/', 32)
    # next(tr_gen)
    # import time
    # i=0
    # total=0
    # t1 = time.time()
    # for a in tr_gen:
    #     t2 = time.time() - t1
    #     # print(a[0].shape)
    #     # print(a[1][0].shape)
    #     # print(a[1][1].shape)
    #     # print(t2)
    #     i += 1
    #     total += t2
    #     time.sleep(0.3)
    #     if i == stop_value:
    #         break
    #     t1 = time.time()
    # print('total time of {} iterations: {}'.format(i, total))
    with Parallel_image_read_transformer(os.path.join(fashion_dataset_path, 'validation85.txt'), 32, class36, attr200, 10) as pargen:
        next(pargen)
    #     time.sleep(45)
    #     i=0
    #     total=0
    #     t1 = time.time()
    #     for a in pargen:
    #         t2 = time.time() - t1
    #         print(a[0].shape)
    #         print(a[1][0].shape)
    #         print(a[1][1].shape)
    #         i+=1
    #         total += t2
    #         time.sleep(0.3)
    #         t1 = time.time()
    #         if i == stop_value:
    #             break
    # print('total time of {} iterations on parralel: {}'.format(i, total))
    # time.sleep(360)
