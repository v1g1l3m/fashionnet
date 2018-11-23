import numpy as np
import os
import glob
import time
from PIL import Image
from random import shuffle, choice
from multiprocessing import Process, sharedctypes, Lock, Pipe, Queue
import threading
import time
import ctypes
from operator import mul
from functools import reduce
import keras
# from keras.applications.vgg19 import preprocess_input
from keras.applications.resnet50 import preprocess_input

class_names = ['Bomber', 'Flannel', 'Button-Down', 'Trunks', 'Culottes', 'Chinos', 'Jeggings', 'Parka', 'Henley',
               'Jersey', 'Poncho', 'Sweatshorts', 'Cutoffs', 'Coat', 'Kimono', 'Sweatpants', 'Hoodie', 'Joggers',
               'Leggings', 'Jumpsuit', 'Jeans', 'Blazer', 'Romper', 'Top', 'Jacket', 'Sweater', 'Cardigan', 'Skirt',
               'Tank', 'Shorts', 'Blouse', 'Tee', 'Dress']


class Parallel_image_transformer(object):
    def __init__(self, path, input_shape):
        self.path = os.path.split(path)[0]
        self.batch_size = input_shape[0]
        self.img_width = input_shape[2]
        self.img_height = input_shape[1]
        self.img_name_class_attr_bbox_part = []
        with open(path) as f:
            for line in f:
                line = line.split()
                img_path = line[0].replace('\\', '/')
                # bboxattr = np.zeros((4,))  # segmentation
                # if line[1] == '0.0-0.0-0.0':
                #     bboxattr[0] = 1.
                # else:
                #     for i, x in enumerate([float(y) for y in line[1].split('-')]):
                #         bboxattr[1+i] = x
                bboxattr = np.zeros((524,))  # classification
                attr = [int(x) for x in line[1].split('-')]
                for i in attr:
                    bboxattr[i] = 1
                bboxattr[491 + class_names.index(line[2])] = 1
                self.img_name_class_attr_bbox_part.append((img_path, bboxattr))
        self.lockr = threading.Lock();
        self.lockr.acquire()
        self.lockw = threading.Lock();
        self.lockw.acquire()
        self.p = threading.Thread(target=self.write_to_queue)
        self.p.start()

    def __iter__(self):
        return self

    def write_to_queue(self):
        while True:
            shuffle(self.img_name_class_attr_bbox_part)
            images_list = []
            bboxattr_list = []
            for img_path, bboxattr in self.img_name_class_attr_bbox_part:
                current_size = len(images_list)
                img = Image.open(img_path)
                img = img.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
                img = np.array(img, dtype=np.float32)
                if len(img.shape) < 3 or img.shape[2] != 3:
                    continue
                images_list.append(img)
                bboxattr_list.append(bboxattr)
                if (current_size < self.batch_size - 1):
                    continue
                self.memx = preprocess_input(np.array(images_list, dtype=np.float32))
                self.memy = np.array(bboxattr_list, dtype=np.float32)
                images_list = []
                bboxattr_list = []
                self.lockr.release()
                self.lockw.acquire()

    def next(self):
        self.lockr.acquire()
        # ret = (self.memx, self.memy)  # segmentation
        ret = (self.memx, [self.memy[:, :491], self.memy[:, 491:]])  # classification
        self.lockw.release()
        return ret

    def terminate(self):
        self.p.join(1)

    def __next__(self):
        return self.next()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()


class Parallel_np_arrays_reader(object):

    def __init__(self, path, out_keys, maxsize=30, numproc=3):
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
                    self.q.put((temp['btl'], temp[self.out_keys[0]][:, :491]))
                else:
                    self.q.put((temp['btl'], [temp[x] for x in self.out_keys]))

    def next(self):
        ret = self.q.get()
        return ret

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