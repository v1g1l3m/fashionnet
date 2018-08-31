import numpy as np
import os
import glob
from skimage.io import imread
from PIL import Image
from random import shuffle
from multiprocessing import Process, Queue
import time

# GLOBALS
img_width = 224             # For VGG16
img_height = 224            # For VGG16
img_channel = 3

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

    def __init__(self, path, maxzise=100):
        self.q = Queue(maxzise)
        self.path = path
        self.p = Process(target=self.write_to_queue)
        self.p.start()

    def __iter__(self):
        return self

    def write_to_queue(self):
        while True:
            with open(self.path) as f:
                for btl_name in f:
                    temp = np.load(open(btl_name.rstrip(), 'rb'))
                    self.q.put((temp['btl'], [temp['cls'], temp['attr']]))

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

class Parallel_image_read_transformer(object):

    def __init__(self, dir, batch_size, maxzise=100):
        self.batch_size = batch_size
        self.q = Queue(maxzise)
        class_names = []
        for name in sorted(os.listdir(dir)):
            if os.path.isdir(os.path.join(dir, name)):
                class_names.append(name)
        self.img_name_class_iou_tuples = []
        for class_name in class_names:
            dataset_path = os.path.join(dir, class_name)
            images_path_name = sorted(glob.glob(dataset_path + '/*.jpg'))
            for name in images_path_name:
                if os.name == 'nt':
                    name = name.replace('\\', '/')
                iou = np.float(name.split('_')[-1].split('.jpg')[0])
                class_1_hot = np.zeros((len(class_names),), dtype=np.float32)
                class_1_hot[class_names.index(name.split('/')[-2])] = iou
                self.img_name_class_iou_tuples.append((name, class_1_hot, iou))
        self.p = Process(target=self.write_to_queue)
        self.p.start()

    def __iter__(self):
        return self

    def write_to_queue(self):
        while True:
            shuffle(self.img_name_class_iou_tuples)
            images_list = []
            class_1_hot_list = []
            iou_list = []
            for path, class_1_hot, iou in self.img_name_class_iou_tuples:
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
                if current_batch_size < self.batch_size - 1:
                    continue
                out_img = np.array(images_list)
                out_cls = np.array(class_1_hot_list)
                out_iou = np.array(iou_list)
                images_list = []
                class_1_hot_list = []
                iou_list = []
                self.q.put((out_img, out_cls))

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
    stop_value = 100
    btl_path = 'bottleneck200/'
    # train_generator = generate_arrays_from_file('bottleneck/btl_train.txt', 10, class_names, (224,224))
    # train_generator = np_arrays_reader(btl_path+'/btl_train_npz.txt')
    # tr_gen = image_transformer('../Data/dataset350/train/', 32)
    # next(tr_gen)
    tr_gen = image_read_transformer('../Data/dataset350/train/', 32)
    next(tr_gen)
    import time
    i=0
    total=0
    t1 = time.time()
    for a in tr_gen:
        t2 = time.time() - t1
        # print(a[0].shape)
        # print(a[1][0].shape)
        # print(a[1][1].shape)
        # print(t2)
        i += 1
        total += t2
        time.sleep(0.3)
        if i == stop_value:
            break
        t1 = time.time()
    print('total time of {} iterations: {}'.format(i, total))
    with Parallel_image_read_transformer('../Data/dataset350/train/', 32, 10) as pargen:
        next(pargen)
        time.sleep(45)
        i=0
        total=0
        t1 = time.time()
        for a in pargen:
            t2 = time.time() - t1
            print(a[0].shape)
            print(a[1][0].shape)
            print(a[1][1].shape)
            i+=1
            total += t2
            time.sleep(0.3)
            t1 = time.time()
            if i == stop_value:
                break
    print('total time of {} iterations on parralel: {}'.format(i, total))
    time.sleep(360)
