import time
import os
import numpy as np
import glob
from random import randint, shuffle
from PIL import Image
from keras import Model
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from utils import init_globals
import logging
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")

### GLOBALS
img_width = 224             # For VGG16
img_height = 224            # For VGG16
img_channel = 3

fashion_dataset_path = '/media/star/3C4C65AA4C65601E/dev/deepfashion/fashion_data/'
btl_path = os.path.join(fashion_dataset_path, 'bottleneck226_350')
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

def save_bottleneck(num_per_file):
    img_name_class_attr_bbox_part = []
    for train_val in ['validation', 'train']:
        with open(os.path.join(fashion_dataset_path, train_val + '_95-5.txt')) as f:
            for line in f:
                line = line.split()
                img_path = line[0]
                img_bbox = [0., 0., 0., 0.]
                if line[1] != 'None':
                    img_bbox = [float(x) for x in line[1].split('-')]
                    img_bbox = (img_bbox[0] + (img_bbox[2]-img_bbox[0])/2,
                                img_bbox[1] + (img_bbox[3]-img_bbox[1])/2,
                                (img_bbox[2]-img_bbox[0]),
                                (img_bbox[3]-img_bbox[1]))
                attrs_1_hot = np.zeros(350, dtype=np.float32)
                if line[2].split('-')[0] != 'None':
                    for x in map(int, line[2].split('-')):
                        attrs_1_hot[x] = 1
                class_1_hot = np.zeros((len(class_names),), dtype=np.float32)
                pc = 0
                if line[3] != 'None':
                    pc = 1
                    class_1_hot[class_names.index(line[3])] = 1
                img_part = train_val
                img_name_class_attr_bbox_part.append((img_path, pc, img_bbox, class_1_hot, attrs_1_hot, img_part))
                            
    shuffle(img_name_class_attr_bbox_part)
     ## Build the VGG16 network
    model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    model = Model(inputs=model.input, outputs=model.layers[14].output)
    for train_val in ['validation', 'train']:
        images_list = []
        class_1_hot_list = []
        attrs_1_hot_list = []
        bbox_list = []
        # pc_list = []
        index = 0
        with open(os.path.join(btl_path, 'btl_' + train_val + '_npz.txt'), 'w') as fw:
            for img_path, pc, img_bbox, class_1_hot, attrs_1_hot, part in img_name_class_attr_bbox_part:
                if part != train_val:
                    continue
                current_size = len(images_list)
                img = Image.open(img_path.replace('\\', '/'))

                # w, h = img.size[0], img.size[1]
                # if w > h:
                #     d = (w - h) // 2
                #     img = img.crop((d, 0, w - d, h))
                #     img_bbox = ((img_bbox[0]*w-d)/h, img_bbox[1], (img_bbox[2]*w)/h, img_bbox[3])
                # else:
                #     d = (h - w) // 2
                #     img = img.crop((0, d, w, h - d))
                #     img_bbox = (img_bbox[0], (img_bbox[1]*h-d)/w, img_bbox[2], (img_bbox[3]*h)/w)

                img = img.resize((img_width, img_height))
                img = np.array(img).astype(np.float32)
                if len(img.shape)<3 or img.shape[2] != 3:
                    continue
                images_list.append(img)
                class_1_hot_list.append(class_1_hot)
                attrs_1_hot_list.append(attrs_1_hot)
                bbox_list.append(img_bbox)
                # pc_list.append(pc)
                if (current_size < num_per_file-1):
                    continue
                images_list = preprocess_input(np.array(images_list))
                class_1_hot_list = np.array(class_1_hot_list)
                attrs_1_hot_list = np.array(attrs_1_hot_list)
                bbox_list = np.array(bbox_list)
                bbox_list[bbox_list < 0] = 0.
                # pc_list = np.array(pc_list)
                bottleneck_features_train_class = model.predict(images_list, batch_size=48)
                btl_save_file_name = train_val + '/btl_' + train_val + '_' + \
                                    str(num_per_file) + '_' + str(index*num_per_file).zfill(7) + '.npz'
                logging.info('btl_save_file_name {}'.format(btl_save_file_name))
                np.savez_compressed(open(os.path.join(btl_path, btl_save_file_name), 'wb'),
                                    btl=bottleneck_features_train_class,
                                    cls=class_1_hot_list,
                                    attr=attrs_1_hot_list,
                                    bbox=bbox_list,
                                    # pc=pc_list
                                    )
                fw.write(str(btl_save_file_name) + '\n')
                images_list = []
                class_1_hot_list = []
                attrs_1_hot_list = []
                bbox_list = []
                # pc_list = []
                index += 1

if __name__ == '__main__':
    global class_names, input_shape, attr_names, attr200, class35
    class_names, input_shape, attr_names = init_globals(fashion_dataset_path)
    create_bottleneck_structure()
    save_bottleneck(226)

    # #-------------------Generate test samples------------
    # img_path_bbox_attr_cls_tuples_list = []
    # for s in ['train85.txt']:
    #     data_class_names = []
    #     data_attr_idx = []
    #     with open(os.path.join(fashion_dataset_path, s)) as f:
    #         for line in f:
    #             line = line.split()
    #             img_path = line[0]
    #             img_gt_bbox = list(map(float, line[1].split('-')))
    #             attrs_1_hot = np.zeros(200, )
    #             if line[2] != 'None':
    #                 attrs_indx = list(map(int, line[2].split('-')))
    #                 for x in attrs_indx:
    #                     if x in attr200:
    #                         data_attr_idx.append(x)
    #                         attrs_1_hot[attr200.index(x)] = 1
    #             class_1_hot = np.zeros((len(class36),), dtype=np.float32)
    #             if line[3] in class36:
    #                 data_class_names.append(line[3])
    #                 class_1_hot[class36.index(line[3])] = 1
    #             # img_path_bbox_attr_cls_tuples_list.append((img_path, img_gt_bbox, attrs_1_hot, class_1_hot))
    #         with open(os.path.join(fashion_dataset_path, 'attr_data_'+s), 'wb') as f:
    #             pickle.dump(data_attr_idx, f)
    #         with open(os.path.join(fashion_dataset_path, 'class_data_'+s), 'wb') as f:
    #             pickle.dump(data_class_names, f)
    # shuffle(img_path_bbox_attr_cls_tuples_list)
    # crop = 30
    # test = set()
    # classes_idx = dict((x, set()) for x in range(len(class36)))
    # attrs_idx = dict((x, set()) for x in range(len(attr200)))
    # count_shared_idx = dict()
    # for i, tup in enumerate(img_path_bbox_attr_cls_tuples_list):
    #     path, bbox, attrs, cls = tup[0], tup[1], tup[2], tup[3]
    #     for x in np.argwhere(attrs==1):
    #         if len(attrs_idx[x[0]]) < crop:
    #             attrs_idx[x[0]].add(i)
    #         # if count_shared_idx.setdefault(i, 0):
    #         #     count_shared_idx[i] += + 1
    #     for x in np.argwhere(cls == 1):
    #         if len(classes_idx[x[0]]) < crop:
    #             classes_idx[x[0]].add(i)
    #         # if count_shared_idx.setdefault(i, 0):
    #         #     count_shared_idx[i] += 1
    # # for a in attrs_idx:
    # #     for c in classes_idx:
    # [test.add(y) for x in attrs_idx.values() for y in x]
    # [test.add(y) for x in classes_idx.values() for y in x]
    # print(len(test))
    # images_list = []
    # class_1_hot_list = []
    # attrs_1_hot_list = []
    # bbox_list = []
    # for idx in test:
    #     tup = img_path_bbox_attr_cls_tuples_list[idx]
    #     path, bbox, attrs, cls = tup[0], tup[1], tup[2], tup[3]
    #     img = Image.open(path)
    #     w, h = img.size[0], img.size[1]
    #     img = img.resize((img_width, img_height))
    #     img = np.array(img).astype(np.float32)
    #     images_list.append(img)
    #     bbox_list.append([bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h, w, h])
    #     attrs_1_hot_list.append(attrs)
    #     class_1_hot_list.append(cls)
    # images_list = preprocess_input(np.array(images_list))
    # bbox_list = np.array(bbox_list)
    # attrs_1_hot_list = np.array(attrs_1_hot_list)
    # class_1_hot_list = np.array(class_1_hot_list)
    # np.savez(open(os.path.join(fashion_dataset_path, 'test%d.npz' % crop), 'wb'), img=images_list,
    #          cls=class_1_hot_list, attr=attrs_1_hot_list, bbwh=bbox_list)
    # model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    # bottleneck_features_train_class = model.predict(images_list)
    # np.savez(open(os.path.join(fashion_dataset_path, 'btl_test%d.npz' % crop), 'wb'), btl=bottleneck_features_train_class,
    #          cls=class_1_hot_list, attr=attrs_1_hot_list, bbwh=bbox_list)

    # -------------------Generate test samples for post model------------
    # crop = 30
    # img_name_class_attr_iou_tuples = []
    # for train_val in ['validation', 'test']:
    #     for class_name in class36:
    #         dataset_class_path = os.path.join(dataset_path, train_val, class_name)
    #         logging.debug('dataset_train_class_path {}'.format(dataset_class_path))
    #         images_path_name = sorted(glob.glob(dataset_class_path + '/*.jpg'))
    #         for name in images_path_name:
    #             if os.name == 'nt':
    #                 name = name.replace('\\', '/')
    #             iou = name.split('_')[-1].split('.jpg')[0]
    #             if (iou == '1.0'):
    #                 img_name_class_attr_iou_tuples.append((name, class_name))
    # print('dataset power: {}'.format(len(img_name_class_attr_iou_tuples)))
    # shuffle(img_name_class_attr_iou_tuples)
    # images_list = []
    # classes_idx = dict((x, 0) for x in class36)
    # for path, cls in img_name_class_attr_iou_tuples:
    #     if classes_idx[cls] < crop:
    #         images_list.append(path)
    #         classes_idx[cls] += 1
    # print('samples: {}'.format(len(images_list)))
    # imgs = []
    # for path in images_list:
    #     img = Image.open(path)
    #     img = img.resize((img_width, img_height))
    #     img = np.array(img).astype(np.float32)
    #     imgs.append(img)
    # images_list = preprocess_input(np.array(imgs))
    # np.savez(open(os.path.join(fashion_dataset_path, 'post_test%d.npz' % crop), 'wb'), img=images_list)
    # model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    # bottleneck_features_train_class = model.predict(images_list)
    # np.savez(open(os.path.join(fashion_dataset_path, 'post_btl_test%d.npz' % crop), 'wb'), btl=bottleneck_features_train_class)
