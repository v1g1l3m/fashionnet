import time
import os
import numpy as np
import glob
from random import randint, shuffle
from PIL import Image
from keras import Model
# from keras.applications.vgg19 import preprocess_input, VGG19
from keras.applications.resnet50 import preprocess_input, ResNet50
from utils import init_globals
import logging
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")

### GLOBALS
img_width = 224
img_height = 224
img_channel = 3

fashion_dataset_path = 'fashion_data/'
btl_path = os.path.join(fashion_dataset_path, 'bottleneck_500')
btl_train_path = os.path.join(btl_path, 'train')
btl_val_path = os.path.join(btl_path, 'validation')
btl_test_path = os.path.join(btl_path, 'test')

class_names = ['Bomber', 'Flannel', 'Button-Down', 'Trunks', 'Culottes', 'Chinos', 'Jeggings', 'Parka', 'Henley', 'Jersey', 'Poncho', 'Sweatshorts', 'Cutoffs', 'Coat', 'Kimono', 'Sweatpants', 'Hoodie', 'Joggers', 'Leggings', 'Jumpsuit', 'Jeans', 'Blazer', 'Romper', 'Top', 'Jacket', 'Sweater', 'Cardigan', 'Skirt', 'Tank', 'Shorts', 'Blouse', 'Tee', 'Dress']

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
                bboxattr = np.zeros((524,))
                attr = [int(x) for x in line[1].split('-')]
                for i in attr:
                    bboxattr[i] = 1
                bboxattr[491 + class_names.index(line[2])] = 1
                img_part = train_val
                img_name_class_attr_bbox_part.append((img_path, bboxattr, img_part))
    shuffle(img_name_class_attr_bbox_part)

    model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    # model = Model(inputs=model.input, outputs=model.layers[16].output)
    for train_val in ['validation', 'train']:
        images_list = []
        attrs_cls_list = []
        index = 0
        with open(os.path.join(btl_path, 'btl_' + train_val + '_npz.txt'), 'w') as fw:
            for img_path, attrs_cls, part in img_name_class_attr_bbox_part:
                if part != train_val:
                    continue
                current_size = len(images_list)
                img = Image.open(img_path.replace('\\', '/'))
                img = img.resize((img_width, img_height), resample=Image.BILINEAR)
                img = np.array(img).astype(np.float32)
                if len(img.shape) < 3 or img.shape[2] != 3:
                    continue
                images_list.append(img)
                attrs_cls_list.append(attrs_cls)
                if (current_size < num_per_file-1):
                    continue
                images_list = preprocess_input(np.array(images_list))
                attrs_cls_list = np.array(attrs_cls_list, dtype=np.float32)

                bottleneck_features_train_class = model.predict(images_list, batch_size=64)
                btl_save_file_name = train_val + '/btl_' + train_val + '_' + \
                                    str(num_per_file) + '_' + str(index*num_per_file).zfill(7) + '.npz'
                np.savez_compressed(open(os.path.join(btl_path, btl_save_file_name), 'wb'),
                                    btl=bottleneck_features_train_class.astype(np.float32),
                                    attr_cls=attrs_cls_list)
                fw.write(str(btl_save_file_name) + '\n')
                images_list = []
                attrs_cls_list = []
                index += 1
                logging.info('{} btl_save_file_name {}'.format(index, btl_save_file_name))

if __name__ == '__main__':
    create_bottleneck_structure()
    save_bottleneck(320)