import numpy as np
import os
import glob
import time
from PIL import Image
from random import shuffle, choice
from multiprocessing import Process, sharedctypes, Lock, Pipe, Queue
import time
import ctypes
from operator import mul
from functools import reduce
import keras
from keras_applications.vgg16 import preprocess_input

class Parallel_image_transformer3(object):
    def __init__(self, path, shapex, shapey, d=None):
        self.path = os.path.split(path)[0]
        self.batch_size = shapex[0]
        self.img_width = shapex[2]
        self.img_height = shapex[1]
        self.img_name_class_attr_bbox_part = []
        with open(path) as f:
            for line in f:
                line = line.split()
                img_path = line[0].replace('\\', '/')
                pcbboxattr = np.zeros((1005,))
                if line[1] != 'None':
                    img_bbox = [float(x) for x in line[1].split('-')]
                    pcbboxattr[1:5] = (img_bbox[0] + (img_bbox[2] - img_bbox[0]) / 2,
                                       img_bbox[1] + (img_bbox[3] - img_bbox[1]) / 2,
                                       (img_bbox[2] - img_bbox[0]) / 2,
                                       (img_bbox[3] - img_bbox[1]) / 2)
                if line[2].split('-')[0] != 'None':
                    for x in map(int, line[2].split('-')):
                        pcbboxattr[x + 5] = 1
                if line[3] != 'None':
                    pcbboxattr[0] = 1
                self.img_name_class_attr_bbox_part.append((img_path, pcbboxattr))
    def next(self):
        images_list = []
        pcbboxattr_list = []
        while len(images_list) < self.batch_size:
            img_path, pcbboxattr = choice(self.img_name_class_attr_bbox_part)
            img = Image.open(img_path)
            w, h = img.size[0], img.size[1]
            if w > h:
                d = (w - h) // 2
                img = img.crop((d, 0, w - d, h))
                pcbboxattr[1:5] = ((pcbboxattr[1] * w - d) / h, pcbboxattr[2], (pcbboxattr[3] * w) / h, pcbboxattr[4])
            else:
                d = (h - w) // 2
                img = img.crop((0, d, w, h - d))
                pcbboxattr[1:5] = (pcbboxattr[1], (pcbboxattr[2] * h - d) / w, pcbboxattr[3], (pcbboxattr[4] * h) / w)
            img = img.resize((self.img_width, self.img_height))
            img = np.array(img).astype(np.float32)
            if len(img.shape) < 3 or img.shape[2] != 3:
                continue
            images_list.append(img)
            pcbboxattr_list.append(pcbboxattr)
        images_list = preprocess_input(np.array(images_list))
        pcbboxattr_list = np.array(pcbboxattr_list)
        return (images_list, pcbboxattr_list)
    def __next__(self):
        return self.next()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pass

class Parallel_image_transformer2(object):
    def __init__(self, path, shapex, shapey, d=None):
        self.path = os.path.split(path)[0]
        self.batch_size = shapex[0]
        self.img_width = shapex[2]
        self.img_height = shapex[1]
        self.wtm = 0
        self.counter = 0
        self.img_name_class_attr_bbox_part = []
        with open(path) as f:
            for line in f:
                line = line.split()
                img_path = line[0].replace('\\', '/')
                bboxattr = np.zeros((1004,))
                if line[1] != 'None':
                    img_bbox = [float(x) for x in line[1].split('-')]
                    bboxattr[:4] = (img_bbox[0] + (img_bbox[2] - img_bbox[0]) / 2,
                                       img_bbox[1] + (img_bbox[3] - img_bbox[1]) / 2,
                                       (img_bbox[2] - img_bbox[0]) / 2,
                                       (img_bbox[3] - img_bbox[1]) / 2)
                if line[2].split('-')[0] != 'None':
                    for x in map(int, line[2].split('-')):
                        bboxattr[x+4] = 1
                self.img_name_class_attr_bbox_part.append((img_path, bboxattr))
        self.shmemx = sharedctypes.RawArray(ctypes.c_double, reduce(mul, shapex))
        self.shapex = shapex
        self.shmemy = sharedctypes.RawArray(ctypes.c_double, reduce(mul, shapey))
        self.shapey = shapey
        self.lockr = Lock(); self.lockr.acquire()
        self.lockw = Lock(); self.lockw.acquire()
        self.p = Process(target=self.write_to_queue)
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
                w, h = img.size[0], img.size[1]
                bboxattr[:4] = (bboxattr[0] / w, bboxattr[1] / h, bboxattr[2] / w, bboxattr[3] / h)
                img = img.resize((self.img_width, self.img_height))
                img = np.array(img).astype(np.float32)
                if len(img.shape) < 3 or img.shape[2] != 3:
                    continue
                images_list.append(img)
                bboxattr_list.append(bboxattr)
                if (current_size < self.batch_size - 1):
                    continue
                images_list = preprocess_input(np.array(images_list))
                bboxattr_list = np.array(bboxattr_list)
                vx = np.ctypeslib.as_array(self.shmemx)
                vx = vx.reshape(self.shapex)
                vx[:] = images_list[:]
                vy = np.ctypeslib.as_array(self.shmemy)
                vy = vy.reshape(self.shapey)
                vy[:] = bboxattr_list[:]
                self.lockr.release()
                images_list = []
                bboxattr_list = []
                self.lockw.acquire()
    def next(self):
        t = time.time()
        self.lockr.acquire()
        vx = np.ctypeslib.as_array(self.shmemx)
        vx = vx.reshape(self.shapex)
        vy = np.ctypeslib.as_array(self.shmemy)
        vy = vy.reshape(self.shapey)
        ret = (np.copy(vx), [np.copy(vy[:,:4]), np.copy(vy[:,4:1004])])
        self.lockw.release()
        self.wtm += time.time() - t
        self.counter += 1
        return ret
    def terminate(self):
        print('Summary time waiting to fetch data: {}'.format(self.wtm))
        if self.counter:
            print ('Average time waiting, ms: {}'.format(self.wtm*1000/self.counter))
        self.p.terminate()
        time.sleep(0.1)
        if not self.p.is_alive():
            self.p.join(timeout=1.0)
    def __next__(self):
        return self.next()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

class Parallel_np_arrays_reader3(object):
    def __init__(self, path, out_keys, c=None, d=None):
        self.path = os.path.split(path)[0]
        self.out_key = out_keys
        self.np_arrays_path_list = []
        self.wtm = 0
        self.counter = 0
        with open(path) as f:
            for btl_name in f:
                self.np_arrays_path_list.append(btl_name.rstrip())
    def next(self):
        temp = np.load(open(os.path.join(self.path, choice(self.np_arrays_path_list)), 'rb'))
        return (temp['btl'], temp[self.out_key])
    def __next__(self):
        return self.next()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pass

class Parallel_np_arrays_reader2(object):
    def __init__(self, path, out_keys, slices, d=None):
        self.path = os.path.split(path)[0]
        self.out_keys = out_keys
        self.slices = slices
        self.np_arrays_path_list = []
        self.wtm = 0
        self.counter = 0
        with open(path) as f:
            for btl_name in f:
                self.np_arrays_path_list.append(btl_name.rstrip())
        temp = np.load(open(os.path.join(self.path, btl_name.rstrip()), 'rb'))
        x = temp['btl']
        self.shmemx = sharedctypes.RawArray(ctypes.c_double, len(x.reshape(-1)))
        self.shapex = x.shape
        y = np.concatenate([temp[x] for x in self.out_keys], axis=-1)
        self.shmemy = sharedctypes.RawArray(ctypes.c_double, len(y.reshape(-1)))
        self.shapey = y.shape
        self.lockr = Lock(); self.lockr.acquire()
        self.lockw = Lock(); self.lockw.acquire()
        self.p = Process(target=self.write_to_queue)
        self.p.start()
    def __iter__(self):
        return self
    def write_to_queue(self):
        while True:
            shuffle(self.np_arrays_path_list)
            for btl_name in self.np_arrays_path_list:
                temp = np.load(open(os.path.join(self.path, btl_name), 'rb'))
                vx = np.ctypeslib.as_array(self.shmemx)
                vx = vx.reshape(self.shapex)
                vx[:] = temp['btl'][:]
                vy = np.ctypeslib.as_array(self.shmemy)
                vy = vy.reshape(self.shapey)
                vy[:] = np.concatenate([temp[x] for x in self.out_keys], axis=-1)[:]
                self.lockr.release()
                self.lockw.acquire()
    def next(self):
        t = time.time()
        self.lockr.acquire()
        vx = np.ctypeslib.as_array(self.shmemx)
        vx = vx.reshape(self.shapex)
        vy = np.ctypeslib.as_array(self.shmemy)
        vy = vy.reshape(self.shapey)
        ret = (np.copy(vx), [np.copy(vy[:,s]) for s in self.slices])
        self.lockw.release()
        self.wtm += time.time() - t
        self.counter += 1
        return ret
    def terminate(self):
        print('Summary time waiting to fetch data: {}'.format(self.wtm))
        if self.counter:
            print ('Average time waiting, ms: {}'.format(self.wtm*1000/self.counter))
        self.p.terminate()
        time.sleep(0.1)
        if not self.p.is_alive():
            self.p.join(timeout=1.0)
    def __next__(self):
        return self.next()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

class Parallel_np_arrays_reader1(object):

    def __init__(self, path, out_keys, maxsize=30, numproc=3):
        self.q = Queue(maxsize)
        self.path = os.path.split(path)[0]
        self.out_keys = out_keys
        self.np_arrays_path_list = []
        self.wtm = 0
        self.counter = 0
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
                    self.q.put((temp['btl'], temp[self.out_keys[0]]))
                else:
                    self.q.put((temp['btl'], [temp[x] for x in self.out_keys]))
                
    def next(self):
        t = time.time()
        ret = self.q.get()
        self.wtm += time.time() - t
        self.counter += 1
        return ret

    def terminate(self):
        print('Summary time waiting to fetch data: {}'.format(self.wtm))
        if self.counter:
            print ('Average time waiting, ms: {}'.format(self.wtm*1000/self.counter))
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

if __name__ == "__main__":
    wtm = time.time()
    train_steps = 100
    with Parallel_image_transformer2('/media/star/3C4C65AA4C65601E/dev/deepfashion/fashion_data/train_95-5.txt', (32,224,224,3), (32,1005)) as train_gen:
    # with Parallel_np_arrays_reader3('/media/star/3C4C65AA4C65601E/dev/deepfashion/fashion_data/bottleneck128/btl_train_npz.txt', 'pcbboxattr', 30, 3) as train_gen:
    # with Parallel_np_arrays_reader2('/media/star/3C4C65AA4C65601E/dev/deepfashion/fashion_data/bottleneck128/btl_train_npz.txt', 'pcbboxattr') as train_gen:
    # train_gen = np_arrays_reader('/media/star/3C4C65AA4C65601E/dev/deepfashion/fashion_data/bottleneck128/btl_train_npz.txt', 'pcbboxattr')
        for i in range(train_steps):
            x, y = next(train_gen)
            print(y[0][:5])
    wtm = time.time() - wtm
    print ('Total time : {}'.format(wtm))
            # time.sleep(0.03)

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
    # with Parallel_image_read_transformer(os.path.join(fashion_dataset_path, 'validation85.txt'), 32, class35, attr200, 10) as pargen:
    #     next(pargen)
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
