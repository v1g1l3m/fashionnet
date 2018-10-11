import os
import sys
import datetime
import time
import pickle
import functools
from PIL import Image
from keras import backend as K
from keras.initializers import VarianceScaling
from keras.optimizers import *
from keras.models import Model
from keras.models import model_from_json, load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, TerminateOnNaN
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from sklearn.utils import class_weight
from generator import *
from utils import init_globals, plot_history, get_validation_data
import logging
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")

### GLOBALS
img_width = 224             # For VGG16
img_height = 224            # For VGG16
img_channel = 3
fashion_dataset_path = '/media/star/3C4C65AA4C65601E/dev/deepfashion/fashion_data/'
# fashion_dataset_path = 'fashion_data/'
btl_path = os.path.join(fashion_dataset_path, 'bottleneck226_350')


def create_model(is_input_bottleneck, input_shape, param):
    if is_input_bottleneck is True:
        model_inputs = Input(shape=(input_shape), name='input_vgg16')
        common_inputs = model_inputs
    else:
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        model_inputs = base_model.input
        common_inputs = base_model.output

    x = MaxPooling2D(padding='same', pool_size=(3, 3))(common_inputs)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    fc_local = Dense(200, activation='relu', name='fc_local')(x)

    x = Conv2D(512, name='conv_5_1', kernel_size=[3, 3], data_format='channels_last', activation='relu', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='uniform'))(common_inputs)
    x = Conv2D(512, name='conv_5_2', kernel_size=[3, 3], data_format='channels_last', activation='relu', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='uniform'))(x)
    x = Conv2D(3072, name='conv_5_3', kernel_size=[3, 3], data_format='channels_last', activation='relu', kernel_initializer=VarianceScaling(mode='fan_avg', distribution='uniform'))(x)
    cnv_pool = GlobalAveragePooling2D()(x)

    x = Concatenate()([cnv_pool, fc_local])
    x = BatchNormalization()(x)
    head_attr = Dense(350, activation='sigmoid', name='pr_attr')(x)
    head_class = Dense(len(class_names), activation='softmax', name='pr_cls')(x)

    x = BatchNormalization()(fc_local)
    head_bbox = Dense(4, activation='sigmoid', name='pr_bbox')(x)

    ## Create Model
    model = Model(inputs=model_inputs, outputs=[head_bbox, head_attr, head_class])
    # if is_input_bottleneck is False:
    for layer in model.layers:
        if layer.name in ['pr_bbox', 'fc_local']:
            logging.info('Not taining layer: {}'.format(layer.name)) 
            layer.trainable = False
    logging.info('summary:{}'.format(model.summary()))
    return model

def loss_cross(y_true, y_pred):
    y_true_present = y_true[..., 0:1]
    loss = y_true_present * K.mean(K.binary_crossentropy(y_true[..., 5:], y_pred[..., 5:]), axis=-1) + \
           y_true_present * K.mean(K.square(y_pred[..., 1:5] - y_true[..., 1:5]), axis=-1) + \
           K.binary_crossentropy(y_true_present, y_pred[..., 0:1])
    return loss

def loss_cat_cross(y_true, y_pred):
    y_true_present = y_true[..., 0:1]
    loss = y_true_present * K.categorical_crossentropy(y_true[..., 5:], y_pred[..., 5:]) + \
           y_true_present * K.mean(K.square(y_pred[..., 1:5] - y_true[..., 1:5]), axis=-1) + \
           K.binary_crossentropy(y_true_present, y_pred[..., 0:1])
    return loss

def loss_mse(y_true, y_pred):
    y_true_present = y_true[..., 0:1]
    loss = y_true_present * K.mean(K.square(y_pred[..., 1:] - y_true[..., 1:]), axis=-1) + \
           K.binary_crossentropy(y_true_present, y_pred[..., 0:1])
    return loss

import keras.losses
keras.losses.loss_cat_cross = loss_cat_cross

def train_model(output_path, param):
    # batch_size = 52
    # val_steps = 12717//batch_size
    # train_steps = 241597//batch_size
    train_steps = 1525#762#1525
    val_steps = 79
    epochs = 50
    # with Parallel_image_transformer2('/media/star/3C4C65AA4C65601E/dev/deepfashion/fashion_data/train_95-5.txt', (batch_size, 224, 224, 3), (batch_size, 1004)) as train_gen:
    #     with Parallel_image_transformer2('/media/star/3C4C65AA4C65601E/dev/deepfashion/fashion_data/validation_95-5.txt', (batch_size, 224, 224, 3), (batch_size, 1004)) as val_gen:
    with Parallel_np_arrays_reader1(os.path.join(btl_path, 'btl_train_npz.txt'), ['attr', 'cls'], 10) as train_gen:
        with Parallel_np_arrays_reader1(os.path.join(btl_path, 'btl_validation_npz.txt'), ['attr', 'cls'], 10) as val_gen:
    # with Parallel_np_arrays_reader2(os.path.join(btl_path, 'btl_train_npz.txt'), ['bbox', 'attr', 'cls'], [slice(0, 4), slice(4, 354),slice(354, 400)]) as train_gen:
    #     with Parallel_np_arrays_reader2(os.path.join(btl_path, 'btl_validation_npz.txt'), ['bbox', 'attr', 'cls'], [slice(0, 4), slice(4, 354),slice(354, 400)]) as val_gen:
        # temp = get_validation_data(os.path.join(btl_path, 'btl_validation_npz.txt'))
        # val_data = (temp['btl'], [temp['bbox'], temp['attr']])
#         with open(os.path.join(fashion_dataset_path, 'attr_weights.pkl'), 'rb') as f:
#             attr_weight = pickle.load(f)
            log_path = os.path.join(output_path, 'model_train.csv')
            csv_log = CSVLogger(log_path , separator=';', append=False)
            filepath = os.path.join(output_path, "best_model-{epoch:03d}-{loss:.4f}-{val_loss:.4f}.h5")
            # early_stop = EarlyStopping(monitor='acc', patience=10, verbose=1, mode='auto')
            checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False,
                                         save_weights_only=False, mode='auto', period=1)
            # callbacks_list = [csv_log, checkpoint, early_stop]
            callbacks_list = [csv_log, checkpoint, TerminateOnNaN()]

            model = create_model(True, (14, 14, 512), param)
            # model = create_model(False, (224, 224, 3), param)
            # model.load_weights('output2/best_weights.hdf5', by_name=True)
            # model.load_weights('output1/best_weights.hdf5', by_name=True)
            # model = load_model('output2/final_model.h5')
            # model.load_weights('output1/best-model.hdf5', by_name=True)
            # model.save('output2/final_model.h5')
            return
            # with open(os.path.join(output_path, 'bottleneck_fc_model.json'), 'w') as f:
            #     f.write(model.to_json())
            plot_model(model, to_file=os.path.join(output_path, 'model.png'), show_shapes=True, show_layer_names=False)
            ## Compile
            model.compile(# optimizer=param,
                          # optimizer=SGD(lr=3e-6, momentum=0.9, nesterov=True),
                          # optimizer=Adam(lr=1e-3),
                          # optimizer=Adadelta(lr=1e-3),
                          optimizer=RMSprop(lr=2e-4),
                          loss={
                                'pr_attr': 'categorical_crossentropy',
                                'pr_bbox': 'mse',
                                'pr_cls': 'binary_crossentropy'
                                },
                          metrics=['accuracy'])
            t_begin = datetime.datetime.now()
            ## Fit
            model.fit_generator(train_gen, steps_per_epoch=train_steps,
                                    epochs=epochs,
                                    validation_data=val_gen,
                                    validation_steps=val_steps,
                                    # class_weight=[1., 1., 1., 1., 1.] + list(attr_weight),
                                    # use_multiprocessing=True,
                                    callbacks=callbacks_list)

    print(datetime.datetime.now())
    print('total_time: {}'.format(str(datetime.datetime.now() - t_begin)))
    print('model saved to: {}'.format(output_path))
    model.save(os.path.join(output_path, 'final_model.h5'))
    model.save_weights(os.path.join(output_path, 'final_weights.hdf5'))
    plot_history(output_path)
    
def task(param):
    output_path = 'output/'
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
        if len(sys.argv) > 2 and sys.argv[2] == 'plot':
            plot_history(output_path, show=True)
            exit(0)
    else:
        if os.path.exists(output_path):
            i = 1
            while os.path.exists(output_path):
                output_path = 'output%d/' % i
                i += 1
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.info('output path: {}'.format(output_path))
    train_model(output_path, param)

### MAIN ###
if __name__ == '__main__':
    global class_names, input_shape, attr_names
    class_names, input_shape, attr_names = init_globals(fashion_dataset_path)
    logging.info('bottleneck path: {}'.format( btl_path))
    task(1)

