import math
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
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TerminateOnNaN, LearningRateScheduler
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras_contrib.applications import ResNet, basic_block, bottleneck

from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from sklearn.utils import class_weight
from generator import *
from utils import init_globals, plot_history, get_validation_data
import logging
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")

### GLOBALS
fashion_dataset_path = 'fashion_data/'
btl_path = os.path.join(fashion_dataset_path, 'bottleneck_500')


def create_model(is_input_bottleneck, input_shape, param):
    if is_input_bottleneck is True:
        model_inputs = Input(shape=(input_shape), name='input_cnn')
        common_inputs = model_inputs
    else:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        # base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        model_inputs = base_model.input
        common_inputs = base_model.output
        # common_inputs = base_model.layers[16].output
        # for layer in base_model.layers:
        #     layer.trainable = False


    x = Conv2D(2048, name='conv_5_1', kernel_size=[1, 1], kernel_initializer=VarianceScaling(mode='fan_avg', distribution='uniform'))(common_inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    head_attr = Dense(491, activation='sigmoid', name='pr_attr')(x)

    x = Conv2D(200, name='conv_5_2', kernel_size=[1, 1], kernel_initializer=VarianceScaling(mode='fan_avg', distribution='uniform'))(common_inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    head_cls =  Dense(33, activation='softmax', name='pr_cls')(x)
    ## Create Model
    model = Model(inputs=model_inputs, outputs=[head_attr, head_cls])
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

def step_decay(epoch):
	initial_lrate = 2e-3
	drop = 0.5
	epochs_drop = 1.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def train_model(output_path, param):
    batch_size = int(sys.argv[2])
    val_steps = 11407//batch_size
    train_steps = 216708//batch_size
    # val_steps = 35
    # train_steps = 677
    epochs = 10
    with Parallel_image_transformer('fashion_data/train_95-ac.txt', (batch_size, 224, 224, 3)) as train_gen:
        with Parallel_image_transformer('fashion_data/validation_95-ac.txt', (batch_size, 224, 224, 3)) as val_gen:
    # with Parallel_np_arrays_reader(os.path.join(btl_path, 'btl_train_npz.txt'), ['attr_cls'], 5) as train_gen:
    #     with Parallel_np_arrays_reader(os.path.join(btl_path, 'btl_validation_npz.txt'), ['attr_cls'], 5) as val_gen:
            log_path = os.path.join(output_path, 'model_train.csv')
            csv_log = CSVLogger(log_path , separator=';', append=False)
            filepath = os.path.join(output_path, "best_model-{epoch:03d}-{loss:.4f}-{val_loss:.4f}.h5")
            # early_stopper = EarlyStopping(min_delta=0.001, patience=10)
            checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False,
                                         save_weights_only=False, mode='auto', period=1)
            lrate = LearningRateScheduler(step_decay)
            # callbacks_list = [csv_log, checkpoint, early_stopper]
            callbacks_list = [csv_log, checkpoint, lrate]

            model = create_model(False, (224, 224, 3), param)
            # model = create_model(True, (7, 7, 2048), param)
            # model = ResNet((160, 96, 3), 4, basic_block, repetitions=[2, 2, 2, 1])
            model.load_weights('output3/best_weights.hdf5', by_name=True)
            # model = load_model('output4/best_model-001-1.2537-1.3404.h5')
            for layer in model.layers:
                if layer.name not in {'pr_cls', 'conv_5_2'}:
                    layer.trainable = False
            print(model.summary())
            plot_model(model, to_file=os.path.join(output_path, 'model.png'), show_shapes=True, show_layer_names=False)
            # ## Compile
            model.compile(#optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True),
                          optimizer=Adam(lr=1e-3, decay=1e-5),
                          # optimizer=Adadelta(),
                          # loss='binary_crossentropy',
                          loss={'pr_attr': 'binary_crossentropy', 'pr_cls': 'categorical_crossentropy'},
                          # loss_weights=[1., 0.05],
                          metrics=['categorical_accuracy'])
            t_begin = datetime.datetime.now()
            ## Fit
            model.fit_generator(train_gen, steps_per_epoch=train_steps,
                                    epochs=epochs,
                                    validation_data=val_gen,
                                    validation_steps=val_steps,
                                    verbose=1,
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
            plot_history(output_path)
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
    class_names, class_type, input_shape, attr_names = init_globals(fashion_dataset_path)
    logging.info('bottleneck path: {}'.format( btl_path))
    task(1)

